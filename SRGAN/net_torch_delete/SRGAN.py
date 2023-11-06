import torch
import torch.nn as nn


class SRGAN(nn.Module):
    def __init__(self, generator, discriminator=None, gan_loss=None, pixel_loss=None, perceptual=None):
        super(SRGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.gan_loss = gan_loss
        self.pixel_loss = pixel_loss
        self.perceptual = perceptual

    def forward_train(self, inputs, data_samples=None, **kwargs):
        return self.forward_tensor(inputs, data_samples, training=True, **kwargs)

    def forward_tensor(self, inputs, data_samples=None, training=False):
        feats = self.generator(inputs)
        return feats

    def if_run_g(self):
        return self.step_counter % self.disc_steps == 0 and self.step_counter >= self.disc_init_steps

    def if_run_d(self):
        return self.discriminator and self.gan_loss

    def g_step(self, batch_outputs, batch_gt_data):
        losses = dict()
        if self.pixel_loss:
            losses['loss_pix'] = self.pixel_loss(batch_outputs, batch_gt_data)
        if self.perceptual_loss:
            loss_percep, loss_style = self.perceptual_loss(
                batch_outputs, batch_gt_data)
            if loss_percep is not None:
                losses['loss_perceptual'] = loss_percep
            if loss_style is not None:
                losses['loss_style'] = loss_style
        if self.gan_loss and self.discriminator:
            fake_g_pred = self.discriminator(batch_outputs)
            losses['loss_gan'] = self.gan_loss(fake_g_pred, target_is_real=True, is_disc=False)
        return losses

    def d_step_real(self, batch_outputs, batch_gt_data):
        real_d_pred = self.discriminator(batch_gt_data)
        loss_d_real = self.gan_loss(real_d_pred, target_is_real=True, is_disc=True)
        return loss_d_real

    def d_step_fake(self, batch_outputs, batch_gt_data):
        fake_d_pred = self.discriminator(batch_outputs.detach())
        loss_d_fake = self.gan_loss(fake_d_pred, target_is_real=False, is_disc=True)
        return loss_d_fake

    def g_step_with_optim(self, batch_outputs, batch_gt_data, optim_wrapper):
        g_optim_wrapper = optim_wrapper['generator']
        with g_optim_wrapper.optim_context(self):
            losses_g = self.g_step(batch_outputs, batch_gt_data)
        parsed_losses_g, log_vars_g = self.parse_losses(losses_g)
        g_optim_wrapper.update_params(parsed_losses_g)
        return log_vars_g

    def d_step_with_optim(self, batch_outputs, batch_gt_data, optim_wrapper):
        log_vars = dict()
        d_optim_wrapper = optim_wrapper['discriminator']
        with d_optim_wrapper.optim_context(self):
            loss_d_real = self.d_step_real(batch_outputs, batch_gt_data)
        parsed_losses_dr, log_vars_dr = self.parse_losses(dict(loss_d_real=loss_d_real))
        log_vars.update(log_vars_dr)
        loss_dr = d_optim_wrapper.scale_loss(parsed_losses_dr)
        d_optim_wrapper.backward(loss_dr)
        with d_optim_wrapper.optim_context(self):
            loss_d_fake = self.d_step_fake(batch_outputs, batch_gt_data)
        parsed_losses_df, log_vars_df = self.parse_losses(dict(loss_d_fake=loss_d_fake))
        log_vars.update(log_vars_df)
        loss_df = d_optim_wrapper.scale_loss(parsed_losses_df)
        d_optim_wrapper.backward(loss_df)
        if d_optim_wrapper.should_update():
            d_optim_wrapper.step()
            d_optim_wrapper.zero_grad()
        return log_vars

    def extract_gt_data(self, data_samples):
        batch_gt_data = data_samples.gt_img
        return batch_gt_data

    def train_step(self, data, optim_wrapper) -> Dict[str, torch.Tensor]:
        g_optim_wrapper = optim_wrapper['generator']
        data = self.data_preprocessor(data, True)
        batch_inputs = data['inputs']
        data_samples = data['data_samples']
        batch_gt_data = self.extract_gt_data(data_samples)
        log_vars = dict()
        with g_optim_wrapper.optim_context(self):
            batch_outputs = self.forward_train(batch_inputs, data_samples)
        if self.if_run_g():
            set_requires_grad(self.discriminator, False)
            log_vars_d = self.g_step_with_optim(batch_outputs=batch_outputs, batch_gt_data=batch_gt_data, optim_wrapper=optim_wrapper)
            log_vars.update(log_vars_d)
        if self.if_run_d():
            set_requires_grad(self.discriminator, True)
            for _ in range(self.disc_repeat):
                log_vars_d = self.d_step_with_optim(
                    batch_outputs=batch_outputs.detach(),
                    batch_gt_data=batch_gt_data,
                    optim_wrapper=optim_wrapper)
            log_vars.update(log_vars_d)
        if 'loss' in log_vars:
            log_vars.pop('loss')
        self.step_counter += 1
        return log_vars




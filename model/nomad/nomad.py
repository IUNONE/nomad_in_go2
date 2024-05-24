import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from model.nomad.nomad_gps import NoMaD_GPS, replace_bn_with_gn
from model.nomad.base_module import DenseNetwork
from model.diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # pip install diffusers==0.11.1
from warmup_scheduler import GradualWarmupScheduler

class NoMaD(L.LightningModule):

    def __init__(self, data_params, model_params, train_params):
        super().__init__()

        self.displacement_norm = data_params["displacement_norm"]
        self.deploy_unnorm = data_params["deploy_unnorm"]

        self.vision_encode_param = model_params["vision_encoder"]
        self.noise_pred_net_param = model_params["noise_pred_net"]        
        self.len_traj_pred = model_params["len_traj_pred"]

        self.optimizer_params = train_params["optimizer"]
        self.loss_weight = float(train_params["loss_weight"])
        self.epochs = train_params["max_epochs"]
        self.save_hyperparameters()

        #-------------------------- model --------------------------#

        if self.vision_encode_param["type"] == "nomad_gps":
            self.vision_encoder = NoMaD_GPS(
                obs_encoding_size = self.vision_encode_param["encoding_size"],
                context_size = self.vision_encode_param["context_size"],
                mha_num_attention_heads = self.vision_encode_param["mha_num_attention_heads"],
                mha_num_attention_layers = self.vision_encode_param["mha_num_attention_layers"],
                mha_ff_dim_factor = self.vision_encode_param["mha_ff_dim_factor"],
            )
            # TODO: check here to define whether to use GN or BN
            self.vision_encoder = replace_bn_with_gn(self.vision_encoder)            
        else: 
            raise ValueError(f"Vision encoder {self.vision_encode_param['type']} not supported")

        self.dist_pred_network = DenseNetwork(embedding_dim = self.vision_encode_param['encoding_size'])
        
        self.noise_pred_net = ConditionalUnet1D(
                input_dim = 2,
                global_cond_dim = self.vision_encode_param["encoding_size"],
                down_dims = self.noise_pred_net_param["down_dims"],
                cond_predict_scale = self.noise_pred_net_param["cond_predict_scale"],
            )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.noise_pred_net_param["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        
    def forward(self, batch_obs_images, batch_goal_pos, re_disp_target = None):
        '''
                - batch_obs_images: [B, C*(context+1), H, W]
                - batch_goal_pos: [B, 1, 2]
                - re_disp_target: [B, len_traj_pred, 2]
        '''
        B = batch_obs_images.shape[0]

        # 2.1 vision_encoder
        obsgoal_cond = self.vision_encoder(obs_img = batch_obs_images, goal_pos = batch_goal_pos)
            
        # 2.2 distance head
        dist_pred = self.dist_pred_network(obsgoal_cond)

        # 2.3 diffusion head
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,(B,)).long().to(self.device)
        if re_disp_target is not None:
            # noisy_action: [B, T, 2], timestep: [B], obsgoal_cond: [B, D], D=context_size, default=256
            noise = torch.randn(re_disp_target.shape).to(self.device)
            noisy_action = self.noise_scheduler.add_noise(re_disp_target, noise, timesteps)
            # noise_pred: [B, T, 2]
            noise_pred = self.noise_pred_net(sample=noisy_action, timestep=timesteps, global_cond=obsgoal_cond)
        else:
            naction = torch.randn((B, self.len_traj_pred, 2)).to(self.device)
            self.noise_scheduler.set_timesteps(self.noise_pred_net_param["num_diffusion_iters"])
            # denoise in k steps
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.noise_pred_net(
                    sample=naction,
                    timestep=k,
                    global_cond=obsgoal_cond,
                )
                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
            noise_pred = naction
            noise = None

        return noise_pred, dist_pred, noise

    def configure_optimizers(self):

        # build optimizer
        lr = float(self.optimizer_params["lr"])
        self.optimizer_params["optimizer"] = self.optimizer_params["optimizer"].lower()
        if self.optimizer_params["optimizer"] == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.98))
        elif self.optimizer_params["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        elif self.optimizer_params["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Optimizer {self.optimizer_params['optimizer']} not supported")

        # build lr scheduler
        scheduler_params = self.optimizer_params["scheduler"]
        if scheduler_params['type'] is not None: 
            if scheduler_params['type'] == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( 
                    optimizer, 
                    T_max = self.epochs
                )
            elif scheduler_params['type'] == "cyclic":
                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=lr / 10.,
                    max_lr=lr,
                    step_size_up=scheduler_params["cyclic_period"] // 2,
                    cycle_momentum=False,
                )
            elif scheduler_params['type'] == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=scheduler_params["plateau_factor"],
                    patience=scheduler_params["plateau_patience"],
                    verbose=True,
                    )
            else:
                raise ValueError(f"Scheduler {scheduler_params['type']} not supported")

            if scheduler_params["warmup"]:
                print("Using warmup scheduler")
                scheduler = GradualWarmupScheduler(
                    optimizer,
                    multiplier=1,
                    total_epoch=scheduler_params["warmup_epochs"],
                    after_scheduler=scheduler,
                )
        else:
            scheduler = None

        return {
                "optimizer": optimizer, 
                "lr_scheduler": scheduler, 
            } 
    
    def training_step(self, batch, batch_idx):
        '''
            forward model and return loss
            
            - Batch from dataset:
                obs_image : [B, C*(context+1), H, W]
                goal_pos : [B, 1, 2]
                future_waypoints : [B, len_traj_pred, 2]
                distance : [B, 1]
            
            - Loss:
                diffusion_loss
                dist_loss
        '''

        obs_image, goal_pos, future_waypoints, distance = batch
        
        # when training, add noise to the target relative displacement
        # tranform to relative displacement and normalize to [-1,1]
        future_waypoints = torch.cat([torch.zeros_like(future_waypoints)[:, :1, :], future_waypoints], dim=1)
        relative_displacement = future_waypoints[:,1:] - future_waypoints[ :,:-1] 
        relative_displacement = (relative_displacement - self.displacement_norm[0]) / (self.displacement_norm[1] - self.displacement_norm[0]) * 2 -1

        #------------------------------------------------

        noise_pred, dist_pred, noise = self(obs_image, goal_pos, relative_displacement)

        #------------------------------------------------
        # diffusion noise loss
        # TODO: what if we use L2 loss for traj ( in fact, a linear transformation of noise loss )         
        noise_loss = F.mse_loss(noise_pred, noise, reduction="none")
        while noise_loss.dim() > 1:
            noise_loss = noise_loss.mean(dim=-1)
        diffusion_loss = noise_loss.mean()
        # dist loss
        dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance.float()).mean()
        # total loss
        loss = self.loss_weight * dist_loss + ( 1 - self.loss_weight ) * diffusion_loss
        #------------------------------------------------
        loss_log = {"loss": loss, "diffusion_loss": diffusion_loss, "distance_loss": dist_loss} 
        self.log_dict(loss_log)
        print(f" [ Train BATCH {batch_idx} ] loss: {loss:.4f}, diffusion_loss: {diffusion_loss:.4f}, distance_loss: {dist_loss:.4f}")

        return loss

    def validation_step(self, batch, batch_idx):
        
        #-------------------- same as training_step --------------------
        obs_image, goal_pos, future_waypoints, distance = batch
        
        future_waypoints = torch.cat([torch.zeros_like(future_waypoints)[:, :1, :], future_waypoints], dim=1)
        relative_displacement = future_waypoints[:,1:] - future_waypoints[ :,:-1]
        relative_displacement = (relative_displacement - self.displacement_norm[0]) / (self.displacement_norm[1] - self.displacement_norm[0]) * 2 -1
        noise_pred, dist_pred, noise = self(obs_image, goal_pos, relative_displacement)    
        noise_loss = F.mse_loss(noise_pred, noise, reduction="none")
        while noise_loss.dim() > 1:
            noise_loss = noise_loss.mean(dim=-1)
        diffusion_loss = noise_loss.mean()
        dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance.float()).mean()
        loss = self.loss_weight * dist_loss + ( 1 - self.loss_weight ) * diffusion_loss
        loss_log = {
            "val_loss": loss, 
            "val_diffusion_loss": diffusion_loss, 
            "val_distance_loss": dist_loss
        } 
        self.log_dict(loss_log)

        return loss

    def test_step(self, batch, batch_idx):
        '''
            TODO: what the difference between training_step and test_step?
        '''
        obs_image, goal_pos, future_waypoints, distance = batch
        
        # when training, add noise to the target relative displacement

        # 1. tranform to relative displacement
        future_waypoints = torch.cat([torch.zeros_like(future_waypoints)[:, :1, :], future_waypoints], dim=1)
        relative_displacement = future_waypoints[:,1:] - future_waypoints[ :,:-1]
        
        # 2. normalize to [-1,1]
        relative_displacement = (relative_displacement - self.displacement_norm[0]) / (self.displacement_norm[1] - self.displacement_norm[0]) * 2 -1

        #------------------------------------------------

        noise_pred, dist_pred, noise = self(obs_image, goal_pos, relative_displacement)
        
        #------------------------------------------------
        # diffusion noise loss        
        noise_loss = F.mse_loss(noise_pred, noise, reduction="none")
        while noise_loss.dim() > 1:
            noise_loss = noise_loss.mean(dim=-1)
        diffusion_loss = noise_loss.mean()
        # dist loss
        dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance.float()).mean()
        # total loss
        loss = self.loss_weight * dist_loss + ( 1 - self.loss_weight ) * diffusion_loss

        # re-forward to get prediction with no traget input 
        noise_pred, dist_pred, _ = self(obs_image, goal_pos)
        noise_pred = noise_pred.detach().cpu().numpy()
        noise_pred = (noise_pred + 1) / 2 * (np.array(self.deploy_unnorm['max']) - np.array(self.deploy_unnorm['min'])) + np.array(self.deploy_unnorm['min'])
        waypoints = np.cumsum(noise_pred, axis=1)
        traj_L2 = np.linalg.norm(waypoints - future_waypoints.detach().cpu().numpy()[:,1:], axis=2).mean()

        #------------------------------------------------
        test_log = {
            "test_loss": loss, 
            "test_diffusion_loss": diffusion_loss, 
            "test_distance_loss": dist_loss,
            "traj_L2": traj_L2,
        } 
        self.log_dict(test_log)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
    # predictions = trainer.predict(model, data_loader)

        obs_image, goal_pos, _ , _ = batch
         
        noise_pred, dist_pred, _ = self(obs_image, goal_pos)
        noise_pred = noise_pred.detach().cpu().numpy()
        
        # unnorm & cumsum
        noise_pred = (noise_pred + 1) / 2 * (self.deploy_unnorm['max'] - self.deploy_unnorm['min']) + self.deploy_unnorm['min']
        waypoints = np.cumsum(noise_pred, axis=1)
        
        return waypoints, dist_pred
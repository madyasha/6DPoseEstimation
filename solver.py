import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from model import model_parser
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from tensorboardcolab import TensorBoardColab
import matplotlib.pyplot as plt
from pose_utils import *
import pandas as pd



class Solver():
    def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.config = config

        # do not use dropout if not bayesian mode
        # if not self.config.bayesian:
        #     self.config.dropout_rate = 0.0

        self.model = model_parser(self.config.model, self.config.sum_mode, self.config.dropout_rate,
                                  self.config.bayesian)

        self.print_network(self.model, self.config.model)
        self.model_save_path = 'models_%s' % self.config.model
        self.summary_save_path = 'summary_%s' % self.config.model

        if self.config.pretrained_model:
            self.load_pretrained_model()

        if self.config.sequential_mode:
            self.set_sequential_mode()

    # Inner Functions #
    def set_sequential_mode(self):
        if self.config.sequential_mode == 'model':
            self.model_save_path = 'models/%s/models_%s' % (self.config.sequential_mode, self.config.model)
            self.summary_save_path = 'summaries/%s/summary_%s' % (self.config.sequential_mode, self.config.model)
        elif self.config.sequential_mode == 'fixed_weight':
            self.model_save_path = 'models/%s/models_%d' % (self.config.sequential_mode, int(self.config.fixed_weight))
            self.summary_save_path = 'summaries/%s/summary_%d' % (self.config.sequential_mode, int(self.config.fixed_weight))
        elif self.config.sequential_mode == 'batch_size':
            self.model_save_path = 'models/%s/models_%d' % (self.config.sequential_mode, self.config.batch_size)
            self.summary_save_path = 'summaries/%s/summary_%d' % (self.config.sequential_mode, self.config.batch_size)
        elif self.config.sequential_mode == 'learning_rate':
            self.model_save_path = 'models/%s/models_%f' % (self.config.sequential_mode, self.config.lr)
            self.summary_save_path = 'summaries/%s/summary_%f' % (self.config.sequential_mode, self.config.lr)
        elif self.config.sequential_mode == 'beta':
            self.model_save_path = 'models/%s/models_%d' % (self.config.sequential_mode, self.config.beta)
            self.summary_save_path = 'summaries/%s/summary_%d' % (self.config.sequential_mode, self.config.beta)
        else:
            assert 'Unvalid sequential mode'
        print (self.summary_save_path)

    def load_pretrained_model(self):
        model_path = self.model_save_path + '/%s_net.pth' % self.config.pretrained_model
        self.model.load_state_dict(torch.load(model_path))
        print('Load pretrained network: ', model_path)

    def print_network(self, model, name):
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()

        print('*' * 20)
        print(name)
        #print(model)
        print('*' * 20)

    def loss_func(self, input, target):
        diff = torch.norm(input-target, dim=1)
        diff = torch.mean(diff)
        return diff

    def train(self):

      #initializers for plotting stuff
        training_loss = []
        pos_tr_loss = []
        rot_tr_loss = []
        iter_no = []
        val_loss =[]
        pos_val_loss = []
        rot_val_loss = []
        iter_val =[]


        #self.tb = TensorBoardColab()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.config.lr,
                               weight_decay=0.0005)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.config.num_epochs_decay, gamma=0.1)

        num_epochs = self.config.num_epochs

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # Setup for tensorboard
        use_tensorboard = self.config.use_tensorboard
        if use_tensorboard:
            if not os.path.exists(self.summary_save_path):
                os.makedirs(self.summary_save_path)
            writer = SummaryWriter(log_dir=self.summary_save_path)
            

        since = time.time()
        n_iter = 0
        n_iter_val = 0

        # For pretrained network
        start_epoch = 0
        if self.config.pretrained_model:
            start_epoch = int(self.config.pretrained_model)

        # Pre-define variables to get the best model
        best_train_loss = 10000
        best_val_loss = 10000
        best_train_model = None
        best_val_model = None

        for epoch in range(start_epoch, num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs-1))
            print('-'*20)

            error_train = []
            error_val = []

            for phase in ['train', 'val']:

                if phase == 'train':
                    scheduler.step()
                    self.model.train()
                else:
                    self.model.eval()

                data_loader = self.data_loader[phase]

                for i, (inputs, poses) in enumerate(data_loader):

                    inputs = inputs.to(device)
                    #print ('inputs', inputs)
                    poses = poses.to(device)
                    #print ('poses', poses)

                    # Zero the parameter gradient
                    optimizer.zero_grad()

                    # forward
                    pos_out, ori_out = self.model(inputs)
                    #print ('pos_out', pos_out)

                    pos_true = poses[:, :3]
                    #print ('pos true', pos_true)
                    ori_true = poses[:, 3:]

                    beta = self.config.beta
                    ori_out = F.normalize(ori_out, p=2, dim=1)
                    ori_true = F.normalize(ori_true, p=2, dim=1)

                    loss_pos = F.mse_loss(pos_out, pos_true)
                    loss_ori = F.mse_loss(ori_out, ori_true)

                    # loss_pos = F.l1_loss(pos_out, pos_true)
                    # loss_ori = F.l1_loss(ori_out, ori_true)
                    # loss_pos = self.loss_func(pos_out, pos_true)
                    # loss_ori = self.loss_func(ori_out, ori_true)

                    loss = loss_pos + beta * loss_ori
                    loss_print = loss.item()
                    loss_ori_print = loss_ori.item()
                    loss_pos_print = loss_pos.item()

                    if use_tensorboard:
                        if phase == 'train':
                            error_train.append(loss_print)
                            writer.add_scalar('loss/overall_loss', loss_print, n_iter)
                            writer.add_scalar('loss/position_loss', loss_pos_print, n_iter)
                            writer.add_scalar('loss/rotation_loss', loss_ori_print, n_iter)
                            #self.tb.save_value('loss/overall_loss', loss_print, n_iter)
                        elif phase == 'val':
                            error_val.append(loss_print)
                            val_loss.append(loss_print)
                            pos_val_loss.append(loss_pos_print)
                            rot_val_loss.append(loss_ori_print)
                            iter_val.append(n_iter_val)
                            n_iter_val += 1

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        iter_no.append(n_iter)
                        training_loss.append(loss_print)
                        pos_tr_loss.append(loss_pos_print)
                        rot_tr_loss.append(loss_ori_print)
                        n_iter += 1

                    print('{}th {} Loss: total loss {:.3f} / pos loss {:.3f} / ori loss {:.3f}'.format(i, phase, loss_print, loss_pos_print, loss_ori_print))

            # For each epoch
            # error_train = sum(error_train) / len(error_train)
            # error_val = sum(error_val) / len(error_val)
            error_train_loss = np.median(error_train)
            error_val_loss = np.median(error_val)



            if (epoch+1) % self.config.model_save_step == 0:
                save_filename = self.model_save_path + '/%s_net.pth' % epoch
                # save_path = os.path.join('models', save_filename)
                torch.save(self.model.cpu().state_dict(), save_filename)
                if torch.cuda.is_available():
                    self.model.to(device)

            if error_train_loss < best_train_loss:
                best_train_loss = error_train_loss
                best_train_model = epoch
            if error_val_loss < best_val_loss:
                best_val_loss = error_val_loss
                best_val_model = epoch
                save_filename = self.model_save_path + '/best_net.pth'
                torch.save(self.model.cpu().state_dict(), save_filename)
                if torch.cuda.is_available():
                    self.model.to(device)

            print('Train and Validaion error {} / {}'.format(error_train_loss, error_val_loss))
            print('=' * 40)
            print('=' * 40)

            if use_tensorboard:
                writer.add_scalars('loss/trainval', {'train':error_train_loss,
                                                     'val':error_val_loss}, epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        if self.config.sequential_mode:
            f = open(self.summary_save_path + '/train.csv', 'w')

            f.write('{},{}\n{},{}'.format(best_train_loss, best_train_model,
                                          best_val_loss, best_val_model))
            f.close()
            # return (best_train_loss, best_train_model), (best_val_loss, best_val_model)
        #return (iter_no, iter_val (training_loss,pos_tr_loss,rot_tr_loss),(val_loss,pos_val_loss,rot_val_loss))
        return (iter_no, training_loss, pos_tr_loss, rot_tr_loss,val_loss,pos_val_loss,rot_val_loss,iter_val)
    
    def test(self):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(device)
        self.model.eval()

        if self.config.test_model is None:
            test_model_path = self.model_save_path + '/best_net.pth'
        else:
            test_model_path = self.model_save_path + '/{}_net.pth'.format(self.config.test_model)

        print('Load pretrained model: ', test_model_path)
        self.model.load_state_dict(torch.load(test_model_path))

        total_pos_loss = 0
        total_ori_loss = 0
        pos_loss_arr = []
        ori_loss_arr = []
        true_pose_list = []
        estim_pose_list = []
        pvar = []
        ovar = []
        if self.config.bayesian:
            pred_mean = []
            pred_var = []

        num_data = len(self.data_loader)

        for i, (inputs, poses) in enumerate(self.data_loader):
            print(i)

            inputs = inputs.to(device)

            # forward
            if self.config.bayesian:
                num_bayesian_test = 50
                pos_array = torch.Tensor(num_bayesian_test, 3)
                ori_array = torch.Tensor(num_bayesian_test, 4)

                for i in range(num_bayesian_test):
                    pos_single, ori_single = self.model(inputs)
                    pos_array[i, :] = pos_single
                    ori_array[i, :] = F.normalize(ori_single, p=2, dim=1)

                pose_quat = torch.cat((pos_array, ori_array), 1).detach().cpu().numpy()
                pred_pose, pred_var = fit_gaussian(pose_quat)
                
                pos_var = np.sum(pred_var[:3])
                ori_var = np.sum(pred_var[3:])
                pvar.append(pos_var)
                ovar.append(ori_var)
                print ('pose_var', pos_var)
                print ('ori_var', ori_var)

                pos_out = pred_pose[:3]
                ori_out = pred_pose[3:]
            else:
                pos_out, ori_out = self.model(inputs)
                pos_out = pos_out.squeeze(0).detach().cpu().numpy()
                ori_out = F.normalize(ori_out, p=2, dim=1)
                ori_out = quat_to_euler(ori_out.squeeze(0).detach().cpu().numpy())


            pos_true = poses[:, :3].squeeze(0).numpy()
            ori_true = poses[:, 3:].squeeze(0).numpy()

            ori_true = quat_to_euler(ori_true)
            loss_pos_print = array_dist(pos_out, pos_true)
            loss_ori_print = array_dist(ori_out, ori_true)

            true_pose_list.append(np.hstack((pos_true, ori_true)))
            
            if loss_pos_print < 20:
                estim_pose_list.append(np.hstack((pos_out, ori_out)))


            total_pos_loss += loss_pos_print
            total_ori_loss += loss_ori_print
            pos_loss_arr.append(loss_pos_print)
            ori_loss_arr.append(loss_ori_print)
            df= pd.DataFrame(list(map(list, zip(pos_loss_arr,ori_loss_arr))))
            df.to_csv('errors.csv')


            if self.config.bayesian:
                print('{}th Error: pos error {:.3f} / ori error {:.3f}'.format(i, loss_pos_print, loss_ori_print))
                print('{}th std: pos / ori', pos_var, ori_var)
                #f.write('{},{},{},{}\n'.format(loss_pos_print, loss_ori_print, pos_var, ori_var))

            else:
                print('{}th Error: pos error {:.3f} / ori error {:.3f}'.format(i, loss_pos_print, loss_ori_print))



        # position_error = sum(pos_loss_arr)/len(pos_loss_arr)
        # rotation_error = sum(ori_loss_arr)/len(ori_loss_arr)
        position_error = np.median(pos_loss_arr)
        rotation_error = np.median(ori_loss_arr)

        #print('=' * 20)
        print('Overall median pose errer {:.3f} / {:.3f}'.format(position_error, rotation_error))
        print('Overall average pose errer {:.3f} / {:.3f}'.format(np.mean(pos_loss_arr), np.mean(ori_loss_arr)))
        #f.close()

        if self.config.save_result:
          df = pd.DataFrame(true_pose_list)
          #df= pd.DataFrame(list(map(list, zip(true_pose_list,lst2,lst3))))
          df.to_csv('true.csv')
          df = pd.DataFrame(estim_pose_list)
          df.to_csv('estimated.csv')
          df = pd.DataFrame(pvar)
          df.to_csv('pvar.csv')
          df = pd.DataFrame(ovar)
          df.to_csv('ovar.csv')



        if self.config.sequential_mode:
            f = open(self.summary_save_path + '/test.csv', 'w')
            f.write('{},{}'.format(position_error, rotation_error))
            f.close()
            return position_error, rotation_error
  
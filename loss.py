# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints



class customLoss(nn.Module):
    def __init__(self, use_target_weight):
        super(customLoss, self).__init__()
        self.criterion_reg = nn.MSELoss(size_average=True)
        self.criterion_class1 = nn.Sigmoid()
        self.criterion_class2 = nn.BCELoss()
        self.use_target_weight = use_target_weight
 
    def forward(self, output_reg, output_class, target_reg, target_class, target_weight):
        #output_reg.size=[128,18,2]   [x,y]
        #output_class.size=[128,18,1]
        #target.size=[128,18,2] 

        num_joints = output_reg.size(1)
        class_pred = output_class
        class_gt = target_class
        loss_reg = 0

        #binary classification loss(binary cross entropy)
        exp = self.criterion_class1(class_pred)
        loss_class = self.criterion_class2(exp, class_gt)

        #class 0을 예측했다면 계산되지 않도록 key_points를 0,0으로 바꿈
        #key_points_pred = key_points_pred.masked_fill(exp < 0.5, 0)
        #key_points_gt = key_points_gt.masked_fill(exp < 0.5, 0)
        # but exp.size=[128,18], key_points=[128,18,2]라 안될 가능성 존재
        
        mask = torch.cat((exp, exp), dim=2)
        key_points_pred = output_reg.masked_fill(mask < 0.5, 0)
        key_points_gt = target_reg.masked_fill(mask < 0.5, 0)
        
        #regression loss(MSELoss)
        #이렇게 하는 이유가 있겠지..
        key_points_pred = key_points_pred.split(1,1) #[18][128][1][2]
        key_points_gt = key_points_gt.split(1,1)

        for idx in range(num_joints):
            key_points_pred = key_points_pred[idx].squeeze() #[128][2]
            key_points_gt = key_points_gt[idx].squeeze()
            if self.use_target_weight:
                loss_reg += 0.5 * self.criterion_reg(
                    key_points_pred.mul(target_weight[:, idx]),
                    key_points_gt.mul(target_weight[:, idx])
                )
            else:
                loss_reg += 0.5 * self.criterion_reg(key_points_pred, key_points_gt)
        #계산 안되는 joints 수는 빼고 나눠야 하나? 
        return (loss_reg+loss_class) / num_joints
    
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, A, B):
        # 유클리드 거리 계산
        distance = torch.sqrt(torch.sum((A - B) ** 2))
        return distance
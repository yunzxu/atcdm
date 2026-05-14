
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef,accuracy_score
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    """
	threshold 一般通过sklearn.metrics里面的roc_curve得到，具体不赘述，可以参考其他资料。
	:param threshold: array, shape = [n_thresholds]
	"""
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point
def plot_auc(pred,labels,save_path =None):
    plt.figure(figsize=(10,7),dpi=100)
    ytrue = labels
    
    # 假设y_true是真实标签，y_scores是预测的概率或决策值
    fpr, tpr, thresholds = roc_curve(ytrue, pred)#
    roc_auc = auc(fpr, tpr)
    optimal_threshold, point = Find_Optimal_Cutoff(tpr,fpr,thresholds)
    print('optimal_threshold',optimal_threshold)


    # 绘制ROC曲线
    plt.plot(fpr, tpr, label=' ROC curve (AUC = %0.2f) (Oth= %0.2f)' % (roc_auc,optimal_threshold))
    #plt.plot(fpr2, tpr2, label='ROC curve (AUC = %0.2f)' % roc_auc2)
    plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title(str(suptitle))
    plt.legend(loc="lower right")

    plt.scatter(point[0], point[1], c='red', marker='*',s=100)
    plt.annotate('Oth '+str(round(optimal_threshold,4 )),
                xy=(point[0], point[1]),
                xytext=(point[0] + 0.1, point[1] - 0.1),
                arrowprops=dict(facecolor='black', arrowstyle="->"))
    plt.show()
    plt.savefig(save_path,dpi=70,bbox_inches='tight')



def show_train_curve(root_path,train_name, loss_name = ['aeloss','weighted_nll_loss','klloss','gloss','Disc'],step_num=100,save_path = None,):
    log_dir = root_path+'/source/run/' + train_name

    print('loh_dir:',log_dir)

    # 使用SummaryReader读取文件
    ea = EventAccumulator(log_dir)

    ea.Reload()

    # 获取训练损失数据

    # loss_name = ['aeloss','weighted_nll_loss','klloss','gloss','Disc']

    # loss_name = ['mse','*klloss','loss']
    plt.figure(figsize=(6*len(loss_name),10),dpi=70)
    W =2
    H = len(loss_name)
    print(ea.Tags())
    for i in range(len(loss_name)):
        plt.subplot(W,H,i+1)
        losses = ea.Scalars(loss_name[i])  # 'loss' 是你在记录时使用的标签名

        # 提取步数和损失值
        steps = [entry.step for entry in losses]
        loss_values = [entry.value for entry in losses]


        loss_values = np.array(loss_values)
        print('steps',len(steps))
        print('loss_values',loss_values.shape)
        ##取整
        len_tb = len(loss_values)
        ##判断llen_tb是否过长，当超过200时，需要使用平均step_num,不是则直接使用
        if len_tb > 200:
            lenth = loss_values.shape[0]
            if lenth%step_num != 0:
                loss_values = loss_values[:lenth//step_num*step_num]
                steps = steps[:lenth//step_num*step_num]
            loss_values = loss_values.reshape(-1, step_num)

            # 计算每组的平均值
            loss_values = loss_values.mean(axis=1)

        if len_tb>200: 
            plt.plot(steps[::step_num], loss_values)    
        else: 
            plt.plot(steps, loss_values)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training : %s' % loss_name[i])
        plt.grid(True)
    plt.suptitle(train_name)
    if save_path != None:
        plt.savefig(save_path,dpi=70,bbox_inches='tight')
    else:
        plt.savefig(root_path+train_name+'/train_curve.png',dpi=70,bbox_inches='tight')
    plt.show()



def plot_all_scalars(log_dir,save_dir, W=2,step_num=1):
    '''
    用于绘制所有结果
    '''
    event_accumulator = EventAccumulator(log_dir)

    event_accumulator.Reload()
    scalar_keys = event_accumulator.scalars.Keys()
    #设置布局：
    # W =4
    H = len(scalar_keys)//W +1
    plt.figure(figsize=(5*H,6*W), dpi=300)
    for i, key in enumerate(scalar_keys):
        # plt.subplot(len(scalar_keys), 1, i + 1)
        plt.subplot(W,H,i+1)
        scalar_data = event_accumulator.Scalars(key)
        steps = [entry.step for entry in scalar_data]
        values = [entry.value for entry in scalar_data]

        # Apply smoothing by averaging over step_num steps
        if step_num > 1:
            smoothed_steps = steps[::step_num]
            smoothed_values = []
            for j in range(0, len(values), step_num):
                smoothed_values.append(np.mean(values[j:j + step_num]))
            values = smoothed_values
            steps = smoothed_steps

        plt.plot(steps, values, label=key)
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title(f'Scalar: {key}')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_dir,dpi=70,bbox_inches='tight')


def display_center_slices(volume, title="Center Slices", save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    center_slices = [s // 2 for s in volume.shape]

    axes[0].imshow(volume[center_slices[0], :, :], cmap="gray")
    axes[0].set_title("Axial Slice")
    axes[0].axis("off")

    axes[1].imshow(volume[:, center_slices[1], :], cmap="gray")
    axes[1].set_title("Coronal Slice")
    axes[1].axis("off")

    axes[2].imshow(volume[:, :, center_slices[2]], cmap="gray")
    axes[2].set_title("Sagittal Slice")
    axes[2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


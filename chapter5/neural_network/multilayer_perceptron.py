import numpy as np
import sys
sys.path.append('chapter5/utils/')
from features import prepare_for_training
from hypothesis import sigmoid, sigmoid_gradient
#from utils.features import prepare_for_training
#from utils.hypothesis import sigmoid, sigmoid_gradient
# 一个手写数字识别的多层感知机实现
# 输入层784个神经元，输出层10个神经元，隐藏层25个神经元
# 使用梯度下降法进行训练
# 使用sigmoid函数作为激活函数
# 使用交叉熵作为损失函数
# 使用正则化，正则化系数为0.1
# 使用学习率为0.1的梯度下降法进行训练
# 使用1000次迭代
# 关键在于具体的前向传播和反向传播算法的实现
# 需要编写公式的伪代码，本算法在文件夹的图片里

class MultilayerPerceptron:
    def __init__(self,data,labels,layers,normalize_data =False):
        data_processed = prepare_for_training(data,normalize_data = normalize_data)[0]
        self.data= data_processed
        self.labels= labels
        self.layers= layers #784 25 10
        self.normalize_data= normalize_data
        self.thetas = MultilayerPerceptron.thetas_init(layers)
# 输入层神经元784，来源：784=28*28*1，28*28是图片的分辨率，1是图片的颜色通道，灰度图为1，彩色图为3
# 输出层神经元10，来源：0~9共计10个数字
# 隐藏层神经元25，来源：25是人为定义的，可以根据实际情况进行调整     
  
    def predict(self,data):
        data_processed = prepare_for_training(data,normalize_data = self.normalize_data)[0]
        num_examples = data_processed.shape[0]
        
        predictions = MultilayerPerceptron.feedforward_propagation(data_processed,self.thetas,self.layers)
        
        return np.argmax(predictions,axis=1).reshape((num_examples,1))
        
        
        
    def train(self,max_iterations=1000,alpha=0.1):
        unrolled_theta = MultilayerPerceptron.thetas_unroll(self.thetas)
        
        (optimized_theta,cost_history) = MultilayerPerceptron.gradient_descent(self.data,self.labels,unrolled_theta,self.layers,max_iterations,alpha)
        
        
        self.thetas = MultilayerPerceptron.thetas_roll(optimized_theta,self.layers)
        return self.thetas,cost_history
         
    @staticmethod
    def thetas_init(layers):
        num_layers = len(layers)# 得到层数
        thetas = {}# thetas 是一个字典，是权重矩阵，key是层数，value是权重矩阵
        for layer_index in range(num_layers - 1):
            """
                            会执行两次，得到两组参数矩阵：25*(784+1) , 10*(25+1)
                            第一次：25*(784+1) 
                            第二次：10*(25+1)
                            这里的+1是偏置项，偏置项的个数跟输出神经元个数是一致的
                            偏置项的值是随机生成的，但是要保证偏置项的值尽量小一点，这样可以防止过拟合
            """
            in_count = layers[layer_index]# 得到当前层的神经元个数作为输入
            out_count = layers[layer_index+1]# 得到下一层的神经元个数作为输出
            # w*x+b;w是权重矩阵，x是输入矩阵，b是偏置矩阵，b的个数跟输出神经元个数是一致的
            # 这里需要考虑到偏置项，记住一点偏置的个数跟输出的结果是一致的
            thetas[layer_index] = np.random.rand(out_count,in_count+1)*0.05 #随机进行初始化操作，值尽量小一点
        return thetas
    
    @staticmethod
    def thetas_unroll(thetas):
        num_theta_layers = len(thetas)# 得到thetas层数
        unrolled_theta = np.array([])# 得到一个拉成1维的数组并拼接
        for theta_layer_index in range(num_theta_layers): 
            unrolled_theta = np.hstack((unrolled_theta,thetas[theta_layer_index].flatten()))
        return unrolled_theta
    
    @staticmethod
    def gradient_descent(data,labels,unrolled_theta,layers,max_iterations,alpha):
        # 梯度下降法的核心代码，这里的theta是拉成一维的数组，需要转换成字典的形式，然后再进行梯度下降法的计算
        # 需要的参数有：数据，标签，theta，层数，最大迭代次数，学习率
        optimized_theta = unrolled_theta
        cost_history = []
        
        for _ in range(max_iterations):

            cost = MultilayerPerceptron.cost_function(data,labels,MultilayerPerceptron.thetas_roll(optimized_theta,layers),layers)
            cost_history.append(cost)
            theta_gradient = MultilayerPerceptron.gradient_step(data,labels,optimized_theta,layers)
            optimized_theta = optimized_theta - alpha* theta_gradient
        return optimized_theta,cost_history
            
            
    @staticmethod 
    def gradient_step(data,labels,optimized_theta,layers):
        theta = MultilayerPerceptron.thetas_roll(optimized_theta,layers)
        thetas_rolled_gradients = MultilayerPerceptron.back_propagation(data,labels,theta,layers)
        thetas_unrolled_gradients = MultilayerPerceptron.thetas_unroll(thetas_rolled_gradients)
        return thetas_unrolled_gradients
    
    @staticmethod 
    def back_propagation(data,labels,thetas,layers):
        num_layers = len(layers)
        (num_examples,num_features) = data.shape
        num_label_types = layers[-1]
        # 反向传播函数计算（BP），是梯度下降法的核心
        deltas = {}
        #初始化操作
        for layer_index in range(num_layers -1 ):
            in_count = layers[layer_index]
            out_count = layers[layer_index+1]
            deltas[layer_index] = np.zeros((out_count,in_count+1)) #25*785 10*26
        for example_index in range(num_examples):
            layers_inputs = {}
            layers_activations = {}
            layers_activation = data[example_index,:].reshape((num_features,1))#785*1
            layers_activations[0] = layers_activation
            #逐层计算
            for layer_index in range(num_layers - 1):
                layer_theta = thetas[layer_index] #得到当前权重参数值 25*785   10*26
                layer_input = np.dot(layer_theta,layers_activation) #第一次得到25*1 第二次10*1
                layers_activation = np.vstack((np.array([[1]]),sigmoid(layer_input)))# 这里的1是偏置项，偏置项的个数跟输出神经元个数是一致的
                layers_inputs[layer_index + 1] = layer_input #后一层计算结果
                layers_activations[layer_index + 1] = layers_activation #后一层经过激活函数后的结果
            output_layer_activation = layers_activation[1:,:]
            
            delta = {}
            #标签处理
            bitwise_label = np.zeros((num_label_types,1))
            bitwise_label[labels[example_index][0]] = 1
            #计算输出层和真实值之间的差异
            delta[num_layers - 1] = output_layer_activation - bitwise_label
            
            #遍历循环 L L-1 L-2 ...2
            for layer_index in range(num_layers - 2,0,-1):
                layer_theta = thetas[layer_index]
                next_delta = delta[layer_index+1]
                layer_input = layers_inputs[layer_index]
                layer_input = np.vstack((np.array((1)),layer_input))
                #按照公式进行计算
                delta[layer_index] = np.dot(layer_theta.T,next_delta)*sigmoid_gradient(layer_input)
                #过滤掉偏置参数
                delta[layer_index] = delta[layer_index][1:,:]
            for layer_index in range(num_layers-1):
                layer_delta = np.dot(delta[layer_index+1],layers_activations[layer_index].T)
                deltas[layer_index] = deltas[layer_index] + layer_delta #第一次25*785  第二次10*26
                
        for layer_index in range(num_layers -1):
            # 这里的1/num_examples是正则化系数，这里的系数是人为定义的，可以根据实际情况进行调整
            deltas[layer_index] = deltas[layer_index] * (1/num_examples)
            
        return deltas
            
    @staticmethod        
    def cost_function(data,labels,thetas,layers):
        num_layers = len(layers)
        num_examples = data.shape[0]
        num_labels = layers[-1]# 输出层神经元个数10就是分类的标签数
        # 损失函数计算
        # 前向传播走一次
        predictions = MultilayerPerceptron.feedforward_propagation(data,thetas,layers)
        #制作标签，每一个样本的标签都得是one-hot
        bitwise_labels = np.zeros((num_examples,num_labels))
        for example_index in range(num_examples):
            bitwise_labels[example_index][labels[example_index][0]] = 1
        bit_set_cost = np.sum(np.log(predictions[bitwise_labels == 1]))
        bit_not_set_cost = np.sum(np.log(1-predictions[bitwise_labels == 0]))
        cost = (-1/num_examples) *(bit_set_cost+bit_not_set_cost)
        return cost
                
    @staticmethod        
    def feedforward_propagation(data,thetas,layers):    
        num_layers = len(layers)
        num_examples = data.shape[0]
        in_layer_activation = data
        # 前向传播函数计算
        # 逐层计算，这里只有一个隐层
        for layer_index in range(num_layers - 1):
            theta = thetas[layer_index]
            out_layer_activation = sigmoid(np.dot(in_layer_activation,theta.T))
            # 正常计算完之后是num_examples*25,但是要考虑偏置项 变成num_examples*26
            out_layer_activation = np.hstack((np.ones((num_examples,1)),out_layer_activation))# 这里的1是偏置项，偏置项的个数跟输出神经元个数是一致的
            # 这里的in_layer_activation是下一层的输入，out_layer_activation是当前层的输出，下一次循环的输入就是当前层的输出，当前层的输出就是下一次循环的输入
            in_layer_activation = out_layer_activation
            
        #返回输出层结果,结果中不要偏置项了
        return in_layer_activation[:,1:]
                   
    @staticmethod       
    def thetas_roll(unrolled_thetas,layers):    
        # 将拉成一维的数组还原回去
        num_layers = len(layers)
        thetas = {}
        unrolled_shift = 0
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index+1]
            
            thetas_width = in_count + 1
            thetas_height = out_count
            thetas_volume = thetas_width * thetas_height
            start_index = unrolled_shift
            end_index = unrolled_shift + thetas_volume
            layer_theta_unrolled = unrolled_thetas[start_index:end_index]
            thetas[layer_index] = layer_theta_unrolled.reshape((thetas_height,thetas_width))
            # 这样就还原回去了
            unrolled_shift = unrolled_shift+thetas_volume
            # 这里的unrolled_shift是用来记录当前的偏移量，每次都加上当前层的权重矩阵的长度，这样就保证了下一次的偏移量是正确的
        
        return thetas
        
        
        
        
        
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
            
        
        
    

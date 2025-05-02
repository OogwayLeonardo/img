# img
在线预测（二分类模型）
特征定义：
历史活跃天数
D_{\text{active}} = \text{用户在7.11-7.20期间有行为的天数}
最近登录时间距离：
\Delta t = 21 - \text{用户最后一次登录的日期（如最后一次登录为7.20，则} \Delta t = 1 \text{）}
\bar{N}_{\text{interact}} = \frac{\text{用户7.11-7.20总互动次数}}{D_{\text{active}}}
y_{\text{online}} = 
\begin{cases} 
1 & \text{用户预测日有行为} \\
0 & \text{其他}
\end{cases}
XGBoost二分类目标函数：
\mathcal{L} = \sum_{i=1}^n \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right] + \lambda \|\omega\|^2
其中：
p_i = \frac{1}{1 + e^{-f(x_i)}}
f(x_{i}) 为树模型输出，λ 为正则化系数。
互动数预测（回归模型）
特征定义：
用户对博主的历史互动次数：
N_{u,b} = \sum_{t=7.11}^{7.20} \left( \text{点赞}_{u,b,t} + \text{评论}_{u,b,t} + \text{关注}_{u,b,t} \right)
H_{\text{follow}} = \sum_{t=7.11}^{7.20} \text{关注}_{b,t}
H_{\text{attr}} = \frac{\sum_{t=7.11}^{7.20} \left( \text{点赞}_{b,t} + \text{评论}_{b,t} \right)}{\sum_{t=7.11}^{7.20} \text{观看}_{b,t}}
T_u = \mathop{\text{argmax}}_{h=1}^{24} \sum_{t=7.11}^{7.20} \text{行为次数}_{u,h}
XGBoost回归目标函数：
\mathcal{L} = \sum_{i=1}^n \left( y_i - \hat{y}_i \right)^2 + \lambda \|\omega\|^2
其中
\hat{y}_i = f(x_i)
公式总结
在线概率：
p_{\text{online}} = \sigma\left( \sum_{k=1}^K f_k(x) \right)
Sigmoid函数：
\sigma(z) = \frac{1}{1 + e^{-z}}
互动数预测值
\hat{y}_{u,b} = \sum_{k=1}^K f_k(x)

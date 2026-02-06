# MSE计算与阈值算法分析报告

## 1. MSE（均方误差）的定义

### 1.1 MSE的数学定义

**MSE（Mean Squared Error）**：
```
MSE = (1/n) * Σ(y_true - y_pred)^2
```

**代码实现**：
```python
# anomaly_detector.py 第82-83行
# 计算每个样本的MSE
mse = np.mean(np.power(data - reconstructed, 2), axis=1)
```

**解释**：
- `data`：原始输入数据（115维特征向量）
- `reconstructed`：自编码器重建的数据（115维特征向量）
- `np.power(data - reconstructed, 2)`：计算每个维度的重建误差的平方
- `np.mean(..., axis=1)`：对每个样本的所有维度求平均

**物理意义**：
- MSE表示自编码器重建数据的平均误差
- MSE越小，说明自编码器重建能力越好
- MSE越大，说明自编码器重建能力越差

### 1.2 MSE的分布特征

**正态分布假设**：
- 如果自编码器训练良好，MSE应该符合正态分布
- 大部分样本的MSE应该集中在均值附近
- 离群值（异常）应该较少

**异常检测原理**：
- 自编码器在良性数据上训练，学习良性数据的分布
- 对于良性样本，重建误差应该较小
- 对于攻击样本，重建误差应该较大（因为自编码器没有见过攻击数据）
- 通过设置阈值，区分良性样本和攻击样本

## 2. 异常阈值计算分析

### 2.1 当前阈值算法

**代码实现**：
```python
# anomaly_detector.py 第108-112行
# 计算均值和标准差
mean_mse = np.mean(mse_values)
std_mse = np.std(mse_values)

# 计算阈值
self.tr_threshold = mean_mse + std_mse
```

**阈值含义**：
- `mean_mse`：DSopt数据集上MSE的均值
- `std_mse`：DSopt数据集上MSE的标准差
- `tr_threshold = mean_mse + std_mse`：异常阈值

**判定规则**：
- 如果MSE > tr_threshold，则判定为异常
- 否则判定为良性

### 2.2 为什么DSopt上的初始异常候选数量那么多？

**原因分析**：

1. **阈值设置过高**：
   - 当前阈值：`mean_mse + std_mse`
   - 这意味着约16%的样本会被判定为异常（假设正态分布）
   - 如果MSE分布不符合正态分布，可能导致更多样本被判定为异常

2. **MSE分布不均匀**：
   - 如果MSE分布右偏（长尾），可能导致更多样本被判定为异常
   - 如果MSE分布左偏，可能导致较少样本被判定为异常

3. **自编码器重建能力不足**：
   - 如果自编码器重建能力不足，良性样本的MSE可能较大
   - 导致阈值较高，更多样本被判定为异常

### 2.3 验证建议

**添加MSE分布分析**：
```python
def calculate_anomaly_threshold(self, dsopt_data):
    """
    计算异常阈值 tr*
    """
    print(f"\n{'=' * 60}")
    print("CALCULATING ANOMALY THRESHOLD (tr*)")
    print(f"{'=' * 60}")

    # 计算DSopt上的MSE
    mse_values = self.calculate_reconstruction_error(dsopt_data)

    # 计算均值和标准差
    mean_mse = np.mean(mse_values)
    std_mse = np.std(mse_values)
    
    # 计算阈值
    self.tr_threshold = mean_mse + std_mse
    
    print(f"📊 DSopt MSE statistics:")
    print(f"   Mean: {mean_mse:.6f}")
    print(f"   Std: {std_mse:.6f}")
    print(f"   Calculated threshold tr*: {self.tr_threshold:.6f}")
    
    # 添加MSE分布分析
    print(f"📊 MSE distribution analysis:")
    print(f"   Percentiles:")
    for p in [50, 75, 90, 95, 99]:
        print(f"   {p}th percentile: {np.percentile(mse_values, p):.6f}")
    
    # 计算初始异常候选数量
    anomaly_decisions = (mse_values > tr_threshold).astype(int)
    print(f"📊 Initial anomaly detection on DSopt:")
    print(f"   Total samples: {len(dsopt_data)}")
    print(f"   Initial anomaly candidates: {sum(anomaly_decisions)} ({sum(anomaly_decisions)/len(dsopt_data)*100:.2f}%)")
    
    return self.tr_threshold
```

## 3. 滑动窗口优化分析

### 3.1 为什么滑动窗口总是要设置成最大？

**原因分析**：

1. **初始异常候选数量过多**：
   - 如果DSopt上的初始异常候选数量很多（例如：> 10%）
   - 需要很大的窗口大小才能通过多数投票消除误报
   - 例如：如果有1000个初始异常候选，窗口大小为100时，需要超过50个异常决策才能判定为异常

2. **多数投票机制的局限性**：
   - 窗口大小为N时，需要超过N/2个异常决策才能判定为异常
   - 如果初始异常候选分布较广，需要更大的窗口大小

3. **max_window_size限制**：
   - 当前实现：`max_window_size = min(100, len(dsopt_data))`
   - 如果DSopt数据集很大，窗口大小限制为100
   - 可能无法找到最优窗口大小

### 3.2 改进建议

**增加max_window_size限制**：
```python
# 当前实现
max_window_size = min(100, len(dsopt_data))

# 建议修改
max_window_size = min(500, len(dsopt_data))  # 增加到500
```

**添加早停机制**：
```python
# 如果连续N个窗口大小的FPR变化小于阈值，则停止
prev_fpr = None
stable_count = 0
max_stable_count = 10

for window_size in range(1, max_window_size + 1):
    fpr = self.calculate_fpr_with_window(anomaly_decisions, true_labels, window_size)
    
    if prev_fpr is not None and abs(fpr - prev_fpr) < 0.0001:
        stable_count += 1
        if stable_count >= max_stable_count:
            print(f"✅ FPR stabilized at window size {window_size}")
            break
    else:
        stable_count = 0
    
    prev_fpr = fpr
    
    if fpr == 0.0:
        best_window_size = window_size
        print(f"✅ Found optimal window size ws* = {best_window_size}")
        break
```

## 4. 自编码器模型分析

### 4.1 自编码器训练分析

**训练过程**：
```python
# trainer.py 第150-155行
# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='mse',  # 使用MSE作为损失函数
    metrics=['mae']
)
```

**损失函数**：
- 使用MSE（均方误差）作为损失函数
- 自编码器的目标是学习重建输入数据的能力
- 通过最小化MSE来优化模型参数

### 4.2 自编码器是否有缺陷？

**可能的问题**：

1. **重建能力不足**：
   - 如果自编码器重建能力不足，良性样本的MSE可能较大
   - 导致阈值较高，更多样本被判定为异常

2. **过拟合**：
   - 如果自编码器过拟合训练数据，对新数据的泛化能力较差
   - 可能导致良性样本的MSE较大

3. **训练不充分**：
   - 如果训练轮数不足，模型可能没有充分学习
   - 可能导致重建能力不足

### 4.3 验证建议

**分析训练历史**：
```python
# 添加训练历史分析
print(f"📊 Training history analysis:")
print(f"   Initial train loss: {history_dict['loss'][0]:.6f}")
print(f"   Final train loss: {history_dict['loss'][-1]:.6f}")
print(f"   Initial val loss: {history_dict['val_loss'][0]:.6f}")
print(f"   Final val loss: {history_dict['val_loss'][-1]:.6f}")
print(f"   Best val loss: {best_val_loss:.6f}")
```

**分析重建误差**：
```python
# 在DSopt和DStst上分别计算MSE
dsopt_mse = self.calculate_reconstruction_error(dsopt_data)
dstst_mse = self.calculate_reconstruction_error(dstst_data)

print(f"📊 MSE comparison:")
print(f"   DSopt MSE: mean={np.mean(dsopt_mse):.6f}, std={np.std(dsopt_mse):.6f}")
print(f"   DStst MSE: mean={np.mean(dstst_mse):.6f}, std={np.std(dstst_mse):.6f}")
```

## 5. 问题总结与解决方案

### 5.1 为什么DSopt上的初始异常候选数量那么多？

**可能原因**：
1. **阈值设置过高**：`mean_mse + std_mse`可能过高
2. **MSE分布不均匀**：MSE分布可能右偏，导致更多样本被判定为异常
3. **自编码器重建能力不足**：良性样本的MSE可能较大

**解决方案**：
1. **调整阈值算法**：
   ```python
   # 当前实现
   tr_threshold = mean_mse + std_mse
   
   # 建议修改（更保守的阈值）
   tr_threshold = mean_mse + 2 * std_mse  # 使用2倍标准差
   ```

2. **使用百分位数**：
   ```python
   # 使用95百分位数作为阈值
   tr_threshold = np.percentile(mse_values, 95)
   ```

3. **分析MSE分布**：
   ```python
   # 添加MSE分布分析
   print(f"📊 MSE distribution analysis:")
   print(f"   Skewness: {scipy.stats.skew(mse_values):.4f}")
   print(f"   Kurtosis: {scipy.stats.kurtosis(mse_values):.4f}")
   ```

### 5.2 为什么滑动窗口总是要设置成最大？

**可能原因**：
1. **初始异常候选数量过多**：需要很大的窗口大小才能通过多数投票消除误报
2. **max_window_size限制过小**：当前限制为100，可能无法找到最优窗口大小
3. **多数投票机制的局限性**：窗口大小越大，需要越多的异常决策才能判定为异常

**解决方案**：
1. **增加max_window_size限制**：
   ```python
   max_window_size = min(500, len(dsopt_data))  # 增加到500
   ```

2. **添加早停机制**：
   ```python
   # 如果连续N个窗口大小的FPR变化小于阈值，则停止
   prev_fpr = None
   stable_count = 0
   max_stable_count = 10
   
   for window_size in range(1, max_window_size + 1):
       fpr = self.calculate_fpr_with_window(anomaly_decisions, true_labels, window_size)
       
       if prev_fpr is not None and abs(fpr - prev_fpr) < 0.0001:
           stable_count += 1
           if stable_count >= max_stable_count:
               print(f"✅ FPR stabilized at window size {window_size}")
               break
       else:
           stable_count = 0
       
       prev_fpr = fpr
       
       if fpr == 0.0:
           best_window_size = window_size
           print(f"✅ Found optimal window size ws* = {best_window_size}")
           break
   ```

3. **添加自适应窗口大小**：
   ```python
   # 根据DSopt数据集的异常候选数量自适应调整窗口大小
   initial_anomaly_ratio = sum(anomaly_decisions) / len(anomaly_decisions)
   
   if initial_anomaly_ratio > 0.05:
       # 初始异常候选比例较高，使用更大的窗口大小
       max_window_size = min(500, len(dsopt_data))
   else:
       # 初始异常候选比例较低，使用较小的窗口大小
       max_window_size = min(100, len(dsopt_data))
   ```

### 5.3 自编码器是否有缺陷？

**验证建议**：
1. **分析训练历史**：
   ```python
   # 添加训练历史分析
   print(f"📊 Training history analysis:")
   print(f"   Initial train loss: {history_dict['loss'][0]:.6f}")
   print(f"   Final train loss: {history_dict['loss'][-1]:.6f}")
   print(f"   Initial val loss: {history_dict['val_loss'][0]:.6f}")
   print(f"   Final val loss: {history_dict['val_loss'][-1]:.6f}")
   print(f"   Best val loss: {best_val_loss:.6f}")
   ```

2. **分析重建误差**：
   ```python
   # 在DSopt和DStst上分别计算MSE
   dsopt_mse = self.calculate_reconstruction_error(dsopt_data)
   dstst_mse = self.calculate_reconstruction_error(dstst_data)
   
   print(f"📊 MSE comparison:")
   print(f"   DSopt MSE: mean={np.mean(dsopt_mse):.6f}, std={np.std(dsopt_mse):.6f}")
   print(f"   DStst MSE: mean={np.mean(dstst_mse):.6f}, std={np.std(dstst_mse):.6f}")
   print(f"   MSE ratio: {np.mean(dstst_mse)/np.mean(dsopt_mse):.2f}")
   ```

3. **检查模型性能**：
   ```python
   # 检查模型是否过拟合
   train_loss = history_dict['loss']
   val_loss = history_dict['val_loss']
   
   if val_loss[-1] > train_loss[-1] * 1.2:
       print(f"⚠️ Warning: Model may be overfitting!")
       print(f"   Final train loss: {train_loss[-1]:.6f}")
       print(f"   Final val loss: {val_loss[-1]:.6f}")
       print(f"   Overfitting ratio: {val_loss[-1]/train_loss[-1]:.2f}")
   ```

## 6. 完整的修复建议

### 6.1 短期修复（立即实施）

1. **添加MSE分布分析**：
   ```python
   # 在calculate_anomaly_threshold方法中添加
   print(f"📊 MSE distribution analysis:")
   print(f"   Percentiles:")
   for p in [50, 75, 90, 95, 99]:
       print(f"   {p}th percentile: {np.percentile(mse_values, p):.6f}")
   ```

2. **调整阈值算法**：
   ```python
   # 提供多种阈值算法
   # 方法1：均值+标准差
   tr_threshold_1 = mean_mse + std_mse
   
   # 方法2：均值+2*标准差（更保守）
   tr_threshold_2 = mean_mse + 2 * std_mse
   
   # 方法3：95百分位数
   tr_threshold_3 = np.percentile(mse_values, 95)
   
   # 使用方法3
   self.tr_threshold = tr_threshold_3
   ```

3. **增加max_window_size限制**：
   ```python
   max_window_size = min(500, len(dsopt_data))
   ```

### 6.2 中期改进（逐步实施）

1. **添加早停机制**：
   ```python
   # 如果连续N个窗口大小的FPR变化小于阈值，则停止
   prev_fpr = None
   stable_count = 0
   max_stable_count = 10
   
   for window_size in range(1, max_window_size + 1):
       fpr = self.calculate_fpr_with_window(anomaly_decisions, true_labels, window_size)
       
       if prev_fpr is not None and abs(fpr - prev_fpr) < 0.0001:
           stable_count += 1
           if stable_count >= max_stable_count:
               print(f"✅ FPR stabilized at window size {window_size}")
               break
       else:
           stable_count = 0
       
       prev_fpr = fpr
       
       if fpr == 0.0:
           best_window_size = window_size
           print(f"✅ Found optimal window size ws* = {best_window_size}")
           break
   ```

2. **添加自适应窗口大小**：
   ```python
   # 根据DSopt数据集的异常候选数量自适应调整窗口大小
   initial_anomaly_ratio = sum(anomaly_decisions) / len(anomaly_decisions)
   
   if initial_anomaly_ratio > 0.05:
       # 初始异常候选比例较高，使用更大的窗口大小
       max_window_size = min(500, len(dsopt_data))
   else:
       # 初始异常候选比例较低，使用较小的窗口大小
       max_window_size = min(100, len(dsopt_data))
   ```

### 6.3 长期改进（深入研究）

1. **优化自编码器架构**：
   - 增加编码器层数
   - 增加隐藏层维度
   - 使用不同的激活函数

2. **使用更先进的异常检测算法**：
   - 基于密度的异常检测
   - 孤立森林异常检测
   - 自编码器+分类器组合方法

3. **添加详细的性能分析**：
   - 分析不同攻击类型的检测性能
   - 分析不同攻击类型的MSE分布
   - 分析不同攻击类型的混淆矩阵

## 7. 结论

### 7.1 MSE的定义

**MSE（Mean Squared Error）**：
- 表示自编码器重建数据的平均误差
- 计算公式：`MSE = (1/n) * Σ(y_true - y_pred)^2`
- 物理意义：MSE越小，说明自编码器重建能力越好

### 7.2 异常阈值计算

**当前算法**：
- 阈值 = 均值 + 标准差
- 判定规则：MSE > 阈值，则判定为异常

**为什么DSopt上的初始异常候选数量那么多？**
- 可能原因1：阈值设置过高（`mean_mse + std_mse`）
- 可能原因2：MSE分布不均匀（右偏）
- 可能原因3：自编码器重建能力不足

### 7.3 为什么滑动窗口总是要设置成最大？

**当前算法**：
- 多数投票机制
- 窗口大小为N时，需要超过N/2个异常决策才能判定为异常

**为什么总是要设置成最大？**
- 可能原因1：初始异常候选数量过多
- 可能原因2：max_window_size限制过小（100）
- 可能原因3：多数投票机制的局限性

### 7.4 自编码器是否有缺陷？

**验证建议**：
- 分析训练历史
- 分析重建误差
- 检查模型是否过拟合

## 8. 附录

### 8.1 术语表

- **MSE**：均方误差（Mean Squared Error）
- **tr***：异常阈值（Threshold）
- **ws***：滑动窗口大小（Window Size）
- **FPR**：误报率（False Positive Rate）
- **TPR**：真阳性率（True Positive Rate）
- **DSopt**：优化数据集（Optimization Dataset，全部为良性）
- **DStst**：测试数据集（Test Dataset，包含良性+攻击）

### 8.2 参考文献

1. N-BaIoT数据集：https://archive.ics.uci.edu/ml/datasets/n-baiot
2. 自编码器异常检测：https://arxiv.org/abs/1901.03407
3. 异常检测阈值方法：https://en.wikipedia.org/wiki/Anomaly_detection

---

**报告生成时间**：2026-02-06
**报告版本**：v1.0
**作者**：AI Assistant

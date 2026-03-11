# 项目改动

把原来的llm自动研究改为量化投资模型的自动研究，开始之前要阅读一下项目的目标program.md，了解一下原来项目的工程思维和ai代理的设计理念

## 数据改动
现在新的数据获取方式变为本地，而且数据类型也不一样，请你根据下面的**数据说明和获取方式来**修改原来的`prepare.py`，而且你要对**make_dataloader**这个函数进行修改，实现一个相适配的高效数据加载器

1. **数据说明**:生成数据的代码在`/Users/lio/Desktop/quant/autoresearch/.cursor/general_data.py`
- lob_data — 这是订单簿数据，已经修改为固定100ms间隔的数据，奇隆lob_data.shape: (29374196, 4, 10)，【T,channel,level】T是所有的时间，channel分别代表的是订单簿的askprice,bidprice,ask_notional 和 bid_notional,代表卖价，买价，卖订单总额，买方总额；level代表的是订单簿的档位，我们的数据是固定的10档

- trade_data — 这是处理过后的交易数据，具体的处理方式你可以查看`/Users/lio/Desktop/quant/autoresearch/.cursor/Data_Pipeline/preprocessors/trade_data_process.py`这个文件。最后的trade_data.shape: (29374196, 23)，具体的列的含义：
    'has_trade', ## 是否有真实交易
    'time_since_last', ## 距离上次交易的时间间隔
    'vwap', ## 成交量加权平均价
    'total_volume', ## 总成交量
    'total_notional', ## 总成交额
    'trade_count', ## 交易笔数
    'buy_notional', ## 主动买入金额
    'sell_notional', ## 主动卖出金额
    'price_high', ## 最高价
    'price_low', ## 最低价
    'price_first', ## 首笔价格
    'price_last', ## 末笔价格
    'price_std', ## 价格标准差
    'buy_count', ## 买入笔数
    'sell_count', ## 卖出笔数
    'net_flow', ## 净流量
    'net_notional', ## 净成交额
    'trade_imbalance', ## 量失衡
    'notional_imbalance', ## 金额失衡
    'price_range', ## 价格范围
    'price_change',## 价格变动
    'avg_trade_size',## 平均单笔大小
    'trade_count_ratio',## 买卖笔数比

- labels_ret — 这是label的数据，主要是未来1800个时间窗口也就是未来180s的收益预测。具体`/Users/lio/Desktop/quant/autoresearch/.cursor/Data_Pipeline/generators/label_gen.py`。labels_ret.shape: (29374196,)


2. **训练数据获取方式**: 数据获取方式改为本地获取，数据存储在本地数据盘中，直接从本地读取。比如
```
lob_data = np.load('/root/autodl-tmp/train_data/lob_data.npy')`
trade_data = np.load('/root/autodl-tmp/train_data/trade_data.npy')
labels_ret = np.load('/root/autodl-tmp/train_data/trade_labels_ret.npy')
```

3. **评估数据获取方式**: 数据获取方式改为本地获取，数据存储在本地数据盘中，直接从本地读取。比如
```
lob_data = np.load('/root/autodl-tmp/test_data/lob_data.npy')`
trade_data = np.load('/root/autodl-tmp/test_data/trade_data.npy')
labels_ret = np.load('/root/autodl-tmp/test_data/trade_labels_ret.npy')
```
## 模型架构

原来是gpt的模型架构，现在我的模型架构放在`/Users/lio/Desktop/quant/autoresearch/.cursor/Model`这个文件夹中，我使用的模型是`/Users/lio/Desktop/quant/autoresearch/.cursor/Model/multi_modal_transformer.py`这个多模态模型，这就是是baseline的架构，也是第一次启动的model，参数的设置在`/Users/lio/Desktop/quant/autoresearch/.cursor/Model/model_config.yaml`。你可以看到我还有其他的插件和模块没有使用，这一点你可以自己进行研究和改善。

## 评估指标


**评估指标**:原来的评估指标是`prepare.py`的evaluate_bpb函数，但是现在我预测主要是涨跌平这个三分类问题，评估指标暂时设置为loss的大小，越小越好

## 训练改动

1. **训练时间**:原来的训练时间固定为 5 分钟，无需考虑耗时问题。我现在要求要实现3个epoch，而且每个epoch的训练时间不能超过10min，也就是每次的迭代不能超过30min



## 补充说明：
**1.取长补短**要吸取原来项目的一些细节高效的方式，然后适配现在的项目。
**2.尽量完善**在你修改program.md的时候，除了上面的改动，其他的软要求和逻辑尽量和原来的逻辑相同。


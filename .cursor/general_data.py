from pathlib import Path
import numpy as np
save_dir = Path('/root/autodl-tmp/train_data')
save_dir.mkdir(parents=True,exist_ok=True)
#### 特征工程
from Data_Pipeline.preprocessors.lob_data_process import process_lob_data
from Data_Pipeline.preprocessors.trade_data_process import process_trade_data
lob_data_dir = '/root/autodl-tmp/ETHUSDT/20levels_parquet'
trade_data_dir = '/root/autodl-tmp/ETHUSDT/trade'


date = ['2025-11-04','2025-11-05','2025-11-06','2025-11-07','2025-11-08','2025-11-09','2025-11-10',
        '2025-11-11','2025-11-12','2025-11-13','2025-11-14','2025-11-15','2025-11-16','2025-11-17',
        '2025-11-18','2025-11-19','2025-11-20','2025-11-21','2025-11-22','2025-11-23','2025-11-24',
        '2025-11-25','2025-11-26','2025-11-27','2025-11-28','2025-11-29','2025-11-30','2025-12-01',
        '2025-12-02','2025-12-03','2025-12-04','2025-12-05','2025-12-06','2025-12-07',
        ]
# date  = ['2025-12-08','2025-12-09','2025-12-10','2025-12-11','2025-12-12','2025-12-13',
#          '2025-12-14','2025-12-15','2025-12-16','2025-12-17','2025-12-18','2025-12-19',
#          '2025-12-20','2025-12-21','2025-12-22','2025-12-23']
levels = 10


lob_data = process_lob_data(data_dir=lob_data_dir,date = date,levels=levels)
trade_data = process_trade_data(data_dir=trade_data_dir,date = date,window_ms=100)

lob_data.write_parquet(save_dir / 'LOB_ETHUSDT_test.parquet')
trade_data.write_parquet(save_dir / 'Trade_ETHUSDT_test_agg100ms.parquet')




from Data_Pipeline.generators.label_gen import generate_data_dict
lob_data_path = save_dir / 'LOB_ETHUSDT_test.parquet'
trade_data_path = save_dir / 'Trade_ETHUSDT_test_agg100ms.parquet'

label_window = 1800
levels = 10
change_window = None # none 就是对即时的价格进行预测
need_price = True
data_dict,trade_labels_ret,price_data = generate_data_dict(lob_data_path,
                                                                        trade_data_path,
                                                                        levels=levels,
                                                                        label_window=label_window,
                                                                        need_price=need_price)

## 存储为npy格式
np.save(save_dir / 'lob_data.npy',data_dict['lob'])
np.save(save_dir / 'trade_data.npy',data_dict['trade'])
np.save(save_dir / 'trade_labels_ret.npy',trade_labels_ret)
if price_data is not None:
    np.save(save_dir / 'price.npy',price_data)


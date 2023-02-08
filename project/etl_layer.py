import pandas as pd
import os


class ETL:
    def __init__(self):
        self.common_data_path = '../src/raw'
        self.saving_path = '../src/preprocessed'

        self.sales_train = None
        self.shops = None
        self.test = None
        self.item_categories = None
        self.items = None

    def extract(self) -> None:
        self.sales_train = pd.read_csv(f'{self.common_data_path}/sales_train.csv')
        self.shops = pd.read_csv(f'{self.common_data_path}/shops.csv')
        self.test = pd.read_csv(f'{self.common_data_path}/test.csv')
        self.item_categories = pd.read_csv(f'{self.common_data_path}/item_categories.csv')
        self.items = pd.read_csv(f'{self.common_data_path}/items.csv')

    def transform(self) -> None:
        self.sales_train.date = pd.to_datetime(self.sales_train.date)
        self.sales_train['month'] = self.sales_train.date.dt.to_period('M')

        self.sales_train = self.sales_train[self.sales_train.item_cnt_day > -2]
        self.sales_train = self.sales_train[self.sales_train.item_price < 300_000]
        self.sales_train = self.sales_train[self.sales_train.month < '2015-11']

        id_of_duplicated_shops = {10: 11, 0: 57, 1: 58, 40: 39}

        for k, v in id_of_duplicated_shops.items():
            self.shops = self.shops[self.shops.shop_id != k]
            self.sales_train.loc[self.sales_train.shop_id == k, 'shop_id'] = v

        self.test.loc[self.test.shop_id == 10, 'shop_id'] = 11

        self.shops.shop_name = self.shops.shop_name.map(lambda x: x.lstrip('!'))
        self.shops[['city', 'shop_name']] = self.shops['shop_name'].str.split(' ', n=1, expand=True)

        self.sales_train = self.sales_train.drop_duplicates(
            subset=['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day'],
            keep='last')

        self.items.item_name = self.items.item_name.map(lambda x: x.lstrip('!*/'))

        same_items_regardless_case = self.items[self.items.item_name.map(lambda x: x.lower()).duplicated()]
        fully_same = self.items.groupby(self.items.item_name, as_index=False).size().query('size > 1')
        fully_same_item_names = fully_same.item_name.values

        self.items = self.items[self.items.item_id != 12]
        self.sales_train = self.sales_train[self.sales_train.item_id != 12]

        same_items_regardless_case_names = same_items_regardless_case[~same_items_regardless_case.item_name.isin(
            fully_same_item_names)].item_name

        same_items = self.items[self.items.item_name.map(lambda x: x.lower()).isin(
            same_items_regardless_case_names.map(lambda x: x.lower()))]

        same_items_upper_case = same_items[same_items.item_name.str.endswith('(Регион)')]
        same_items_lower_case = same_items[same_items.item_name.str.endswith('(регион)')]

        self.items = self.items[~self.items.item_id.isin(same_items_lower_case.item_id)]

        self.sales_train.loc[self.sales_train.item_id.isin(same_items_lower_case.item_id), 'item_id'] = \
            self.sales_train[self.sales_train.item_id.isin(same_items_lower_case.item_id)].item_id.map(lambda x: x - 1)

        self.test.loc[self.test.item_id.isin(same_items_lower_case.item_id), 'item_id'] = \
            self.test[self.test.item_id.isin(same_items_lower_case.item_id)].item_id.map(lambda x: x - 1)

        self.items = self.items[self.items.item_id != 13012]
        self.sales_train.loc[self.sales_train.item_id == 13012, 'item_id'] = 13011
        self.test.loc[self.test.item_id == 13012, 'item_id'] = 13011

    def load(self):
        self.sales_train = pd.merge(self.sales_train, self.items, on='item_id', how='inner')
        self.sales_train = pd.merge(self.sales_train, self.item_categories, on='item_category_id', how='inner')
        self.sales_train = pd.merge(self.sales_train, self.shops, on='shop_id', how='inner')

        if not os.path.exists(self.saving_path):
            os.mkdir(self.saving_path)

        self.sales_train.to_csv(f'{self.saving_path}/sales_train.csv')
        self.shops.to_csv(f'{self.saving_path}/shops.csv')
        self.test.to_csv(f'{self.saving_path}/test.csv')
        self.item_categories.to_csv(f'{self.saving_path}/item_categories.csv')
        self.items.to_csv(f'{self.saving_path}/items.csv')


if __name__ == '__main__':
    etl = ETL()
    etl.extract()
    etl.transform()
    etl.load()

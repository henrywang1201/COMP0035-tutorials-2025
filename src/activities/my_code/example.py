from pathlib import Path
import matplotlib.pyplot as plt
"""
This script reads data from a CSV file and an Excel file located in the project's 'data' directory.
It demonstrates how to use pathlib for file path management and pandas for data loading.
Functions:
    test: Placeholder function for future testing or demonstration purposes.
Attributes:
    csv_file (Path): Path to the 'paralympics_raw.csv' file.
    xlsx_file (Path): Path to the 'paralympics_all_raw.xlsx' file.
    csv_df (DataFrame): DataFrame containing data from the CSV file.
    xlsx_df_1 (DataFrame): DataFrame containing data from the first sheet of the Excel file.
    xlsx_df_2 (DataFrame): DataFrame containing data from the second sheet of the Excel file.
"""
import pandas as pd



def describe_dataframes(df):
    '''Summary:
    Parameters:
        df: pandas DataFrame
    Returns:
        None
    '''
    # 展示 pandas 数据集的基本信息
    print("--- DataFrame shape (rows, columns) ---")
    print(df.shape)

    print("\n--- First 5 rows ---")
    pd.set_option("display.max_columns", None)
    print(df.head())

    print("\n--- Last 5 rows ---")
    print(df.tail())

    print("\n--- Column labels ---")
    print(df.columns)

    print("\n--- Column datatypes ---")
    print(df.dtypes)

    print("\n--- DataFrame info ---")
    df.info()

    print("\n--- DataFrame describe ---")
    print(df.describe(include='all'))

    # 缺失值统计
    print("\n--- Number of missing values per column ---")
    print(df.isna().sum())

    # 只包含有缺失值的行
    missing_rows = df[df.isna().any(axis=1)]
    print(f"\n--- Rows with any missing values (total: {len(missing_rows)}) ---")
    print(missing_rows)

    # Create a histogram of the DataFrame
    # df.hist()
    columns = ["participants_m", "participants_f"]
    df[columns].hist()

    # Show the plot
    plt.show()


    # 查看 'type' 列的唯一值
    print("Distinct categorical values in the event 'type' column")
    print(csv_df['type'].unique())

    # 统计每个类别出现的次数
    print("\nCount of each distinct categorical value in the event 'type' column")
    print(csv_df['type'].value_counts())

    # 查看 'disabilities_included' 列的唯一值
    print("\nDistinct categorical values in the 'disabilities_included' column")
    print(csv_df['disabilities_included'].unique())

    # 统计每个类别出现的次数
    print("\nCount of each distinct categorical value in the 'disabilities_included' column")
    print(csv_df['disabilities_included'].value_counts())

    # 画所有数值型变量的箱线图
    df.boxplot()
    plt.title("Paralympics Data - Boxplot")
    plt.show()

    # 只画某一列（如 'participants_m'）
    df.boxplot(column='participants_m')
    plt.title("Participants (Male) - Boxplot")
    plt.show()

    # 多列分别画子图
    df[['participants_m', 'participants_f']].plot.box(subplots=True)
    plt.suptitle("Participants by Gender - Boxplot")
    plt.show()


def plot_timeseries(df):
    # 按年份画总参赛人数
    plt.figure(figsize=(10, 6))
    plt.plot(df['start'], df['participants'], marker='o', label='Total Participants')
    plt.xlabel('Year')
    plt.ylabel('Number of Participants')
    plt.title('Paralympics Participants Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 按性别分别画
    plt.figure(figsize=(10, 6))
    plt.plot(df['start'], df['participants_m'], marker='o', label='Male')
    plt.plot(df['start'], df['participants_f'], marker='o', label='Female')
    plt.xlabel('Year')
    plt.ylabel('Number of Participants')
    plt.title('Paralympics Participants by Gender Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def prepare_paralympics_data(raw_df):
    
    # 1. 删除不需要的列
    df_prepared = raw_df.drop(columns=['URL', 'disabilities_included', 'highlights'])
    print("\n--- Columns after dropping ['URL', 'disabilities_included', 'highlights'] ---")
    print(df_prepared.columns)

    # 2. 处理缺失值
    # 删除指定的行（index 0, 17, 31）
    df_prepared = df_prepared.drop(index=[0, 17, 31])
    print("\n--- After dropping rows with missing/irrelevant values (index 0, 17, 31) ---")
    print(df_prepared.head(3))

    # 重新设置索引
    df_prepared = df_prepared.reset_index(drop=True)
    print("\n--- After resetting index ---")
    print(df_prepared.head(3))

    # 4. 修正分类数据
    # type 列全部小写并去除多余空格
    df_prepared['type'] = df_prepared['type'].str.lower().str.strip()
    print("\n--- Unique values in 'type' after cleaning ---")
    print(df_prepared['type'].unique())

    # 5. 更改数据类型
    # 5.1 将部分 float64 列转为 Int64（可容纳缺失值）
    columns_to_change = ['countries', 'events', 'participants_m', 'participants_f', 'participants']
    for col in columns_to_change:
        df_prepared[col] = df_prepared[col].astype('Int64')

    # 5.2 将 start 和 end 列转为 datetime
    df_prepared['start'] = pd.to_datetime(df_prepared['start'], format='%d/%m/%Y')
    df_prepared['end'] = pd.to_datetime(df_prepared['end'], format='%d/%m/%Y')

    # 5.3 打印 start 和 end 列，检查是否有缺失值
    print("\n--- 'start' column unique values ---")
    print(df_prepared['start'].unique())
    print("\n--- 'end' column unique values ---")
    print(df_prepared['end'].unique())

    # 5.4 打印所有列的数据类型
    print("\n--- Data types after conversion ---")
    print(df_prepared.dtypes)

    # 5.5 尝试将剩余 object 类型列转为 string
    for col in df_prepared.select_dtypes(include=['object']).columns:
        df_prepared[col] = df_prepared[col].astype('string')
    print("\n--- Data types after converting object columns to string ---")
    print(df_prepared.dtypes)

    # 6. 添加新计算列 duration（单位：天），插入到 end 列后
    duration_values = (df_prepared['end'] - df_prepared['start']).dt.days.astype('Int64')
    df_prepared.insert(df_prepared.columns.get_loc('end') + 1, 'duration', duration_values)
    print("\n--- 'duration' column (days) ---")
    print(df_prepared['duration'])
    

    # 读取 NPC 代码表，仅保留 Code 和 Name 两列
    npc_codes_path = Path(__file__).parent.parent.joinpath('data', 'npc_codes.csv')
    npc_df = pd.read_csv(npc_codes_path, encoding='utf-8', encoding_errors='ignore', usecols=['Code', 'Name'])

    # 替换 events 数据中的 country 字段为标准名称
    replacement_names = {
        'UK': 'Great Britain',
        'USA': 'United States of America',
        'Korea': 'Republic of Korea',
        'Russia': 'Russian Federation',
        'China': "People's Republic of China"
    }
    df_prepared['country'] = df_prepared['country'].replace(replacement_names)

    # 合并数据，左连接
    merged_df = df_prepared.merge(npc_df, how='left', left_on='country', right_on='Name')

    # 打印 country, Code, Name 列，检查 NaN
    print("\n--- country, Code, Name columns after merge ---")
    print(merged_df[['country', 'Code', 'Name']])

    # 检查是否还有 NaN
    print("\n--- Rows with NaN in Code after merge ---")
    print(merged_df[merged_df['Code'].isna()][['country', 'Code', 'Name']])

    # 删除 Name 列
    merged_df = merged_df.drop(columns=['Name'])

    # 可选：打印合并后部分内容
    print("\n--- Merged DataFrame sample ---")
    print(merged_df.head())


    # 6. 返回或保存清洗后的数据
    return merged_df

if __name__ == "__main__":
    # Define the path to the CSV file using pathlib
    # This script is located in a subfolder so you need to navigate up to the parent (src) and then its parent (project root), then down to the 'data' directory and finally the .csv file
    csv_file = Path(__file__).parent.parent.joinpath('data', 'paralympics_raw.csv')
    xlsx_file = Path(__file__).parent.parent.joinpath('data', 'paralympics_all_raw.xlsx')

    csv_df = pd.read_csv(csv_file)
    xlsx_df_1 = pd.read_excel(xlsx_file,sheet_name=0)
    xlsx_df_2 = pd.read_excel(xlsx_file,sheet_name=1)

    # plot_timeseries(csv_df)


    cleaned_df = prepare_paralympics_data(csv_df)

    output_path = Path(__file__).parent / 'paralympics_cleaned.csv'
    cleaned_df.to_csv(output_path, index=False)
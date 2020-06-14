import pandas as pd
import glob
import os
import seaborn as sns

def check_type(training_directory):
    paths = glob.glob('./' + training_directory + '/*')
    columns_name = ['FileName', 'TypeSepsis', 'Sex', 'Age', 'StartTime', 'LenTime']
    list_rows = []
    type = [0, 1]
    for path in paths:
        file_name = path.split('/')[-1]
        df = pd.read_csv(path, delimiter='|')
        len_time = df.shape[0]
        flag = df[df['SepsisLabel'] == 1].drop_duplicates(subset=['SepsisLabel'])
        age = df['Age'].iloc[0]
        gender = df['Gender'].iloc[0]
        if flag.empty:
            row = [file_name, type[0], gender, age - 1, len_time]
            list_rows.append(row)
        else:
            start = flag.index[0]
            row = [file_name, type[1], gender, age, start, len_time]
            list_rows.append(row)

    save_file = pd.DataFrame(list_rows, columns=columns_name).sort_values(by=['FileName'], ascending=True)
    path_save = './check_setB.csv'
    with open(path_save, 'w') as f:
        save_file.to_csv(f, encoding='utf-8', header=True, index=False)
    print('Complete check type')


def convert_to_csv(training_directory, training_directory_csv):
    paths = glob.glob('./' + training_directory + '/*')
    for path in paths:
        file_name = path.split('/')[-1]
        df = pd.read_csv(path, delimiter='|')
        path_save = './' + training_directory_csv + '/' + file_name
        with open(path_save, 'w') as f:
            df.to_csv(f, encoding='utf-8', header=True, index=False)


#Not useful
def concatenate(training_directory):
    path_folder = './'+ training_directory
    list_files = pd.read_csv('./check.csv')
    num_file = list_files.shape[0]
    for i in range(num_file):
        file = os.path.join(path_folder, list_files.iloc[i]['FileName'])
        df = pd.read_csv(file, delimiter='|')
        if i == 0:
            frames = df
        else:
            frames = [frames, df]
            frames = pd.concat(frames)
    path_save = './summary.csv'
    with open(path_save, 'w') as f:
        frames.to_csv(f, encoding='utf-8', header=True, index=False)

#Not useful
def statistical():
    file_sum = './summary.csv'
    df = pd.read_csv(file_sum)
    name_columns = list(df)
    len = len(name_columns)
    label = name_columns[-1]
    for i in range(len - 1):
        df_temp = df[[name_columns[i], label]]
        df_temp = df_temp.dropna()
        # df_normal = df_temp [df_temp[label]==0].mean()
        # df_sepsis = df_temp [df_temp[label]==1].mean()

        img1 = sns.jointplot(x=name_columns[i], y=label, data=df_temp)
        # img2 = sns.jointplot(x=name_columns[i], y=label, data=df_sepsis)
        # img = df_temp.plot.scatter(x =name_columns[i], y = label, c = 'Red')
        # plt.savefig(img)

        img1.savefig('./visualize/' + str(name_columns[i]) + '.png')
        # img2.savefig('./visualize/' + str(name_columns[i]) + '_sepsis.png')

def main():
    name_dic = 'training_setB'
    name_dic_csv = 'training_setB_csv'
    check_type(name_dic)
    convert_to_csv(name_dic, name_dic_csv)
main()
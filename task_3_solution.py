import pandas as pd

def split_data_into_two_samples(data, random_state = 42):
    check_frame = data.sample(frac=0.3, random_state = 42)
    train_frame = data.drop(check_frame.index)
    return check_frame, train_frame

if __name__ == '__main__':
    data = pd.read_csv('./train.csv')
    check_frame, train_frame = split_data_into_two_samples(data)
    print(check_frame.head(3))
    print(train_frame.head(7))

import pandas as pd
import numpy as np
import sys

class Dataset:

    def __init__(self, data):

        self.df = data


    def preprocessing(self):

        df = pd.DataFrame(self.df)
        df = df.drop('Artist', axis=1).join(df['Artist'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True).rename('Artist'))

        df.reset_index(inplace=True, drop=True)
        df['Artist'] = df['Artist'].astype('string')
        df['first_week_charted'] = pd.to_datetime(df['Weeks Charted'].apply(lambda s: s.split('--')[0]))
        df['last_week_charted'] = pd.to_datetime(df['Weeks Charted'].apply(lambda s: s.split('--')[-1]))
        #set a week of highest rating to a number of week in the current year
        df['Week of Highest Charting'] =\
            pd.to_datetime(df['Week of Highest Charting'].apply(lambda s: s.split('--')[0])).dt.isocalendar().week

        df["Song Name"] = df["Song Name"].astype('string')
        #streams into a int with ',' replacing
        df['Streams'] =  df['Streams'].apply(lambda s: int(s.replace(',','')))
        #dealing with artist's followers. empty sell will be setted to 1, then it will be converted into int datatype,
        #then those values, which became 1, will be setted mean value
        idxs = df[df['Artist Followers'] == ' '].index
        df.loc[idxs, 'Artist Followers'] = 1
        df['Artist Followers'] = df['Artist Followers'].astype(int)
        df.loc[idxs, 'Artist Followers'] = df['Artist Followers'].mean().round().astype(int)
        #songs with missing id get it equal to -1
        df.loc[idxs, 'Artist Followers'] = df['Artist Followers'].mean().round().astype(int)
        df['Song ID'] = df['Song ID'].astype('string')
        #missing genre get -1 value
        id_emp_genre = df[(df['Genre'] == ' ') | (df['Genre'] =='[]')].index
        df.loc[id_emp_genre, ['Genre']] = -1
        df['Genre'] = df['Genre'].astype('string')
        df['Genre'] = df['Genre'].apply(lambda s: s.replace('[', '').replace(']', '').replace('\'', ''))
        df['Genre'] = df['Genre'].astype('string')
        #missing dates in release date will be dropped
        df['Release Date'].replace(' ', np.nan, inplace=True)
        df.dropna(subset=['Release Date'], inplace=True)
        df['Release Date'] = pd.to_datetime(df['Release Date'])
        df.drop(['Weeks Charted'], axis = 1, inplace = True)
        df['Popularity'] = df['Popularity'].astype(int)
        df['Danceability'] = df['Danceability'].astype(float)
        df['Energy'] = df['Energy'].astype(float)
        df['Loudness'] = df['Loudness'].astype(float)
        df['Speechiness'] = df['Speechiness'].astype(float)
        df['Acousticness'] = df['Acousticness'].astype(float)
        df['Liveness'] = df['Liveness'].astype(float)
        df['Tempo'] = df['Tempo'].astype(float)
        df['Duration (ms)'] = df['Duration (ms)'].astype(float)
        df['Valence'] = df['Valence'].astype(float)
    #making chords as numbers
        enc_dict = {}
        for i in range(len(df['Chord'].unique())):
            enc_dict[sorted(df['Chord'].unique())[i]] = i
        df['Chord'] = df['Chord'].apply(lambda s: enc_dict.get(s))
        df['Chord'] = df['Chord'].astype(int)
        transformed_df = df
        return transformed_df

if __name__ == "__main__":
    if len(sys.argv) > 1:

        data = pd.read_csv(str(sys.argv[1]))
        dataset = Dataset(data)

        out_df = dataset.preprocessing()
        out_df.to_csv('out.csv')


    else:
        print('Введите параметр - название')
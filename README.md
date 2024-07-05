# Can K-Means clustering be utilized to group Spotify songs based on the audio features valence and tempo?
![image](images/images/Spotify_Logo.png)
## Table of Contents
1. [Problem Statement](https://github.com/lcar-dsport/clustering_spotify_songs/blob/main/README.md#problem-statement)
2. [The Dataset](https://github.com/lcar-dsport/clustering_spotify_songs/blob/main/README.md#the-dataset)
3. [Executive Summary](https://github.com/lcar-dsport/clustering_spotify_songs/blob/main/README.md#executive-summary)
4. [Data Preprocessing](https://github.com/lcar-dsport/clustering_spotify_songs/blob/main/README.md#data-preprocessing)

## Problem Statement
The goal of this project is to group Spotify songs into clusters based on the audio features tempo and valence using K-Means clustering. An algorithm such as this could be used to create a song recommendation system for Spotify users by recommending songs within the same cluster. 

## The Dataset
This dataset was obtained from Kaggle at: [Kaggle](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs/data). 
The dataset contains the following columns:
- `TRACK_ID`: Song ID
- `TRACK_NAME`: Song Name
- `TRACK_ARTIST`: Song Artist
- `TRACK_POPULARITY`: Song Popularity With a Rating of 0-100 Where a Higher Rating is Better
- `TRACK_ALBUM_ID`: Album ID
- `TRACK_ALBUM_NAME`: Album Name
- `TRACK_ALBUM_RELEASE_DATE`: Album release date dd/mm/yyyy
- `PLAYLIST_NAME`: Name of spotify playlist
- `PLAYLIST_ID`: Playlist ID
- `PLAYLIST_GENRE`: Playlist Genre
- `PLAYLIST_SUBGENRE`: Playlist Subgenre
- `DANCEABILITY`: How suitable a song is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
- `ENERGY`: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity.
- `KEY`: The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation . E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
- `LOUDNESS`: The overall loudness of a track in decibels (dB).
- `MODE`: Major is represented by 1 and minor is 0.
- `SPEECHINESS`: Speechiness detects the presence of spoken words in a track. On a scale of 0.0-1.0, where 1.0 represents a track made up of speech only, and where 0.0 represents a track with no spoken word.
- `ACOUSTICNESS`: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
- `INSTRUMENTALNESS`: A prediction of whether a track contains vocals. A scale of 0.0-1.0, where 1.0 represents the greater likelihood the track contains no vocal content and is mainly instrumental.
- `LIVENESS`: Detects the presence of an audience in the recording. A scale of 0.0-1.0, where 1.0 represents a strong likelihood of the track being live.
- `VALENCE`: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track, where 1.0 means the track is positive and happy.
- `TEMPO`: The tempo of a track in beats per minute (BPM).
- `DURATION_MS`: Duration of a song in milliseconds.


## Executive Summary
This project aimed to investigate whether K-Means clustering can be used to group Spotify songs based on the audio features valence and tempo. This project offers a potential starting point for the development of a future song recommendation system for Spotify users. Due to time constraints and cost/run-time restrictions faced in this project, only two audio features were included in the analysis. Future projects with fewer restrictions should explore whether K-Means clustering can be used to group songs based on all the available audio features for the best and most accurate results.

## Data Preprocessing
### Loading the Data
The file was extracted from Kaggle and loaded into our data warehouse, Exasol. I then connected to the table in Exasol using Jupyter notebooks. 

### Data cleansing
As I was not interested in the playlist data, I dropped any playlist columns.
```
df = df.drop(columns = ['PLAYLIST_NAME','PLAYLIST_ID','PLAYLIST_GENRE','PLAYLIST_SUBGENRE']).copy()
```
I then dropped any duplicate values.
```
df = df.drop_duplicates().copy()
```

```
# data cleansing and transformation
# check for null values
df.isna().sum().sort_values(ascending=False)
# no null values so no cleansing needed yet

# check for duplicate rows 
df.duplicated().any()
# no duplicate rows
```
### Feature Selection
The relevant columns for the clustering algorithm were selected and all other columns were dropped. This was then imported into a new dataframe, `spotify_songs`. 

import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st

client_id = '877c3e1752ee4e98ac1a47ef73921e3f'
client_secret = '4793b0ab150f4d96b1ce3a5e8c969549'

def get_spotify_access_token():
    auth_url = 'https://accounts.spotify.com/api/token'
    response = requests.post(auth_url, data={'grant_type': 'client_credentials'}, auth=(client_id, client_secret))
    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        return None

def get_spotify_link(song_title, token):
    song_title_encoded = song_title.replace(" ", "%20")
    url = f"https://api.spotify.com/v1/search?q={song_title_encoded}&type=track&limit=1"
    headers = {
        'Authorization': f'Bearer {token}'
    }

    try:
        response = requests.get(url, headers=headers)
        response_data = response.json()

        if 'tracks' in response_data and response_data['tracks']['items']:
            track = response_data['tracks']['items'][0]
            track_name = track['name']
            track_url = track['external_urls']['spotify']
            return track_name, track_url
        else:
            return None, "Song not found"
    except Exception as e:
        return None, str(e)

data = pd.read_csv(r'Z:\projects\musicRecommenderSystem\dataset.csv')
data['userRating'] = data['userRating'].apply(lambda x: float(x.split('/')[0]))

label_encoder_genre = LabelEncoder()
label_encoder_album = LabelEncoder()

data['genre'] = label_encoder_genre.fit_transform(data['genre'])
data['album'] = label_encoder_album.fit_transform(data['album'])

X = data[['genre', 'album']]
y = data['userRating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

threshold = 1
accuracy = ((abs(y_pred - y_test) <= threshold).sum() / len(y_test)) * 100

st.set_page_config(page_title="Music Recommendation System", layout="wide", page_icon="🎵")
st.title("🎼 Music Recommendation System")
st.sidebar.title("🎶 Music Recommender")
st.sidebar.write("Choose your preferences to get personalized music recommendations.")

model_type = st.sidebar.selectbox('Choose Recommendation Model', ['Genre-based', 'Artist-based'])
song_input = st.sidebar.selectbox('Select a Song', sorted(data['song'].unique()), help="Select a song to get recommendations based on Genre or Artist")
num_recommendations = st.sidebar.number_input("How many recommendations would you like?", min_value=1, max_value=10, value=4)

selected_song = data[data['song'] == song_input].iloc[0]
genre_input = selected_song['genre']
artist_input = selected_song['singer']

selected_genre = label_encoder_genre.inverse_transform([genre_input])[0]
selected_album = label_encoder_album.inverse_transform([selected_song['album']])[0]

st.subheader("Selected Song Information")
st.write(f"**Song:** {song_input}")
st.write(f"**Genre:** {selected_genre}")
st.write(f"**Album:** {selected_album}")
st.write(f"**Artist(s):** {artist_input}")
st.write(f"**Rating:** {selected_song['userRating']}/10")

spotify_token = get_spotify_access_token()
if spotify_token:
    track_name, spotify_link = get_spotify_link(song_input, spotify_token)
    if track_name:
        st.markdown(f"[Listen on Spotify]({spotify_link})")
    else:
        st.error("Spotify link not found.")
else:
    st.error("Failed to get Spotify token.")

if model_type == 'Genre-based':
    st.subheader(f"🎧 Genre-Based Recommendations")
    filtered_data = data[data['genre'] == genre_input]
    filtered_data = filtered_data[filtered_data['song'] != song_input]

    if len(filtered_data) >= num_recommendations:
        recommended_songs = filtered_data.sample(n=num_recommendations)
    else:
        singer_filtered_data = data[data['singer'] == artist_input]
        singer_filtered_data = singer_filtered_data[singer_filtered_data['song'] != song_input]

        if len(singer_filtered_data) < num_recommendations:
            additional_data = data[data['song'] != song_input]
            combined_data = pd.concat([singer_filtered_data, additional_data]).drop_duplicates()
            recommended_songs = combined_data.nlargest(num_recommendations, 'userRating')
        else:
            recommended_songs = singer_filtered_data.sample(n=num_recommendations)

elif model_type == 'Artist-based':
    st.subheader(f"🎤 Artist-Based Recommendations")
    artist_list = [artist.strip() for artist in artist_input.split(',')] if isinstance(artist_input, str) else []
    artist_filtered_data = data[data['singer'].apply(lambda x: isinstance(x, str) and any(artist in x for artist in artist_list))]
    artist_filtered_data = artist_filtered_data[artist_filtered_data['song'] != song_input]

    if len(artist_filtered_data) < num_recommendations:
        additional_data = data[data['song'] != song_input]
        combined_data = pd.concat([artist_filtered_data, additional_data]).drop_duplicates()
        recommended_songs = combined_data.nlargest(num_recommendations, 'userRating')
    else:
        recommended_songs = artist_filtered_data.sample(n=num_recommendations)

st.subheader(f"🎶 Recommended Songs ({model_type})")
st.write("Here are some songs you might like:")

columns = st.columns(min(len(recommended_songs), 4))

for idx, row in enumerate(recommended_songs.itertuples()):
    track_name, spotify_link = get_spotify_link(row.song, spotify_token)
    with columns[idx % 4]:
        st.markdown("---")
        st.markdown(f"**{row.song}**")
        st.markdown(f"Rating: {row.userRating}/10")
        st.markdown(f"Artist: {row.singer}")
        st.markdown(f"Genre: {label_encoder_genre.inverse_transform([row.genre])[0]}")
        st.markdown(f"Album: {label_encoder_album.inverse_transform([row.album])[0]}")

        if track_name:
            st.markdown(f"[Listen on Spotify]({spotify_link})")

with st.sidebar.expander("📊 Model Performance"):
    st.write(f"**Model Accuracy:** {accuracy:.2f}%")
    st.write(f"**Mean Squared Error:** {mse:.2f}")

st.sidebar.write("Created by Aniket V. Singh")

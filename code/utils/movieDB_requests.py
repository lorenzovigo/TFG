import requests

# API requests to MovieDB needed


def getRequestToken(api_key):
    response = requests.get("https://api.themoviedb.org/3/authentication/token/new?api_key=" + api_key)
    return response.json()["request_token"]


def getMovieDBId(title, year, api_key):
    response = requests.get("https://api.themoviedb.org/3/search/movie?api_key=" + api_key
                            + "&language=en-US&query=" + title + "&year=" + str(year))
    return response.json()["results"][0]["id"]


def getTVDBId(title, year, api_key):
    response = requests.get("https://api.themoviedb.org/3/search/tv?api_key=" + api_key
                            + "&language=en-US&query=" + title + "&year=" + str(year))
    return response.json()["results"][0]["id"]


def getMovieActorIds(movieDBId, api_key):
    response = requests.get("https://api.themoviedb.org/3/movie/" + str(movieDBId) + "/credits?api_key=" + api_key + "&language=en-US")
    ids = []
    for character in response.json()["cast"]:
        ids.append(character["id"])
    return ids


def getTVActorIds(movieDBId, api_key):
    response = requests.get("https://api.themoviedb.org/3/tv/" + str(movieDBId) + "/credits?api_key=" + api_key + "&language=en-US")
    ids = []
    for character in response.json()["cast"]:
        ids.append(character["id"])
    return ids


def getActorName(movieDBId, api_key):
    response = requests.get("https://api.themoviedb.org/3/person/" + str(movieDBId) + "?api_key=" + api_key + "&language=en-US")
    return response.json()['name']

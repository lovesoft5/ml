
import math;

def ReadData():
    file_user_movie = 'F:/机器学习/untitled/ml-100k/u.data'
    file_movie_info = 'F:/机器学习/untitled/ml-100k/u.item'
    user_movie = {}  #存放用户对电影的评分信息
    for line  in open(file_user_movie,encoding='utf-8'):
        user,item,score = line.split('\t')[0:3]
        user_movie.setdefault(user,{})
        user_movie[user][item] = int(score)
    movies = {}#存放电影的基本信息
    for line in open(file_movie_info,encoding='utf-8'):
         (movieId,movieTitle) = line.split('|')[0:2]#前两个元素分别为电影ID，电影名称
         movies[movieId] = movieTitle
    return user_movie,movies

def ItemSimilarity(user_movie):
    C = {}
    N ={}
    for user,items in user_movie.items():
        for i in items.keys():
            N.setdefault(i,0)
            N[i] += 1
            C.setdefault(i,{})
            for j in items.keys():
                if i == j : continue
                C[i].setdefault(j,0)
                C[i][j] += 1
        W = {}
        for i,related_items in C.items():
           W.setdefault(i,{})
           for j,cij in related_items.items():
             W[i][j] = cij/(math.sqrt(N[i] * N[j]))
    return  W

def Recommend(user,user_movie,W,K,N):
    rank = {}
    action_item = user_movie[user]
    for item,score in action_item.items():
        for j,wj in sorted(W[item].items(),key=lambda x:x[1],reverse =True)[0:K]:
            if j in action_item.keys():
                continue
            rank.setdefault(j,0)
            rank[j] +=score * wj
    return dict(sorted(rank.items(),key = lambda x:x[1],reverse=True)[0:N])

if __name__ == "__main__":
    user_movie,movies = ReadData()

    W = ItemSimilarity(user_movie)

    result = Recommend('1',user_movie,W,10,10)

    for i,rating in result:
        print('film: %s,rating: %s' %(movies[i-1],rating))
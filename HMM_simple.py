states = ('Rainy','Sunny')

obervations = ('walk','shop','clean')

start_probability = {'Rainy':0.6,'Sunny':0.4}

trainsition_probability = {
    'Rainy':{'Rainy':0.7,'Sunny':0.3},
    'Sunny':{'Rainy':0.4,'Sunny':0.6}
}

emission_probability = {
    'Rainy':{'walk':0.1,'shop':0.4,'clean':0.5},
    'Sunny':{'walk':0.6,'shop':0.3,'clean':0.1},
}

#打印路径probabity表
def print_dptable(v):
    for i in range(len(v)):
        print (" %7d"% i)
    for y in v[0].keys():
        print("%.5s:"% y)
        for t in range(len(v)):
            print("%0.7s"%("%f" % v[t][y]))


def viterbi(obs,states,start_p,trans_p,emit_p):
    """

        :param obs:观测序列
        :param states:隐状态
        :param start_p:初始概率（隐状态）
        :param trans_p:转移概率（隐状态）
        :param emit_p: 发射概率 （隐状态表现为显状态的概率）
        :return:
    """
    v = [{}]
    path = {}

    for y in states:
        v[0][y] = start_p[y]*emit_p[y][obs[0]]
        path[y] = [y]
    for t in range(1,len(obs)):
        v.append({})
        newpath = {}

        for y in states:
            (prob, state) = max([(v[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states])
            v[t][y] = prob
            newpath[y] = path[state] + [y]
            #print("newpath[y]", newpath[y])
        path = newpath
    print_dptable(v)
    (prob, state) = max([(v[len(obs) - 1][y], y) for y in states])
    return (prob, path[state])

def example():
    return viterbi(obervations,states,start_probability,trainsition_probability,emission_probability)
print(example())
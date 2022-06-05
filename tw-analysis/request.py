import os
import sys
import time
import requests
import json
import re
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import japanize_matplotlib
import itertools
from collections import Counter
import networkx as nx
from libs.LogManager import log
import urllib.parse as urlp
import urllib.request as urlreq
from datetime import datetime as dt
from datetime import timedelta as td
from datetime import timezone as tz
from dotenv import load_dotenv
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer

def main():
    logger = log.getMyLogger(__name__)
    # logger.info('info')

    # .envファイルの内容を読み込見込む
    load_dotenv()

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # 　検索パラメータ
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    keyword = 'ウクライナ'
    # 日本時間
    start_time = '2022-06-01 00:00:00'
    # end_time = '2022-05-19 15:59:59'

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # 　１．Twitterからツイート内容を取得
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    tweet_list = []
    tknzr = Tokenizer()
    delimiter = '_/_'
    next_token = ''
    res_count = 0
    # 1,000件以上は取得しない
    max_count = 1000
    while res_count < max_count : # 取得データが{max_count}件以上になったら、中止する
        try:
            res['meta']['next_token']
        except KeyError: # 次ページがない（next_tokenが存在しない）場合はループを抜ける
            break
        except NameError: # １回目（resが存在しない）
            next_token = ''
        else: # ２回目以降
            next_token = 'next_token='+res['meta']['next_token']+'&'
        finally:
            tweet_fields = 'id,created_at,text,author_id'
            user_fields = 'id,created_at,name,username'
            search_url = 'https://api.twitter.com/2/tweets/search/recent?query={}%20lang%3Aja&start_time={}&max_results=100&sort_order=recency&{}expansions=author_id&tweet.fields={}&user.fields={}'.format(
                urlp.quote(keyword.replace('AND ', ''), encoding='utf-8'), toStrUTC(dt.strptime(start_time, '%Y-%m-%d %H:%M:%S')), next_token, tweet_fields, user_fields
            )
            # print(search_url)
            headers = {}
            headers["Authorization"] = f"Bearer {os.environ['BEARER_TOKEN']}"
            headers["User-Agent"] = "v2RecentSearchPython"
            # 100件ずつ取得
            res = getRequest(search_url, headers, logger)

            # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            # 　２．ツイートを形態素に分ける
            # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
            tweet_list.extend(getWordList(res['data'], delimiter, tknzr))

            res_count += res['meta']['result_count']

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # 　３．単語を数える（棒グラフ）
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    # 除外キーワード
    with open(os.path.join('tw-analysis','stop_words.txt'),'r',encoding='utf-8') as f:
        stop_list = f.read().split('\n')
    stop_list.extend(re.sub('(AND )|(OR )','',keyword).split())

    vec = CountVectorizer(token_pattern='(?u)(.+?){}'.format(delimiter), min_df=3, lowercase=False, encoding='utf-8', decode_error='replace', stop_words=stop_list, analyzer='word')
    # 特徴量の抽出
    matrix = vec.fit_transform(tweet_list)
    vocabulary = vec.get_feature_names_out()
    word_count = np.sum(matrix.toarray(), axis=0)
    # ds = pd.Series({tag: count for tag, count in zip(vocabulary, word_count)})
    ds = pd.Series(data=word_count, index=vocabulary)
    # inplace=True：もとのSeriesも変更
    ds.sort_values(ascending=False, axis=0, inplace=True)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # 　４．単語のペアと頻度を抽出（共起ネットワーク）
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    # 文単位の名詞ペアリストを生成（１行の中から２つの単語を取得する組合せ）
    noun_list = [re.sub('(_/_)+','_/_',t[:-3]).split(delimiter) for t in tweet_list]
    noun_list = [a for a in noun_list if a != '']
    pair_list = [
                list(itertools.combinations(n, 2))
                for n in noun_list if len(noun_list) >=2
                ]

    # 名詞ペアリストの１次元化
    all_pairs = []
    for u in pair_list:
        all_pairs.extend(u)

    # 名詞ペアの頻度をカウント
    cnt_pairs = Counter(all_pairs)

    # データフレームの作成
    noun_1 = []
    noun_2 = []
    frequency = []
    for n,f in cnt_pairs.items():
        if n[0] == n[1] or n[0] in stop_list or n[1] in stop_list:
            continue
        noun_1.append(n[0])
        noun_2.append(n[1])
        frequency.append(f)

    df = pd.DataFrame({'単語ペア１': noun_1, '単語ペア２': noun_2, '出現頻度': frequency})
    df.sort_values(by=['出現頻度'], ascending=False, axis=0, inplace=True)

    # 重み付きデータの設定
    weighted_edges = np.array(df[:50])

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # 　５．グラフ化し、画像へ保存
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

    # グラフオブジェクトの生成
    G = nx.Graph()

    # 重み付きデータの読み込み
    G.add_weighted_edges_from(weighted_edges)

    color = []
    n = G.number_of_nodes()
    for i in range(n):
        color.append((i/n/2+0.4, i/n/2+0.4, 1.0, 0.4))

    sizes = [50*ds[node] if node in ds else 25 for node, deg in G.degree()]

    pos = nx.spring_layout(G, k=0.7)

    # 共起ネットワーク
    plt.figure(figsize=(10,10))
    nodes = nx.draw_networkx_nodes(G, pos, node_shape = 'o', node_color=color, node_size=sizes)
    nodes.set_edgecolor('#777777')
    nx.draw_networkx_edges(G, pos, edge_color=color, width=3, alpha=0.4)
    nx.draw_networkx_labels(G, pos, font_color='k', font_size=14, font_family='MS Gothic', font_weight='bold')
    plt.savefig(os.path.join('images','networkx.png'))

    # plt.show()

    # 棒グラフ
    plt.rcParams['font.family'] = "MS Gothic"
    plt.figure(figsize=[10.0, 5.8])
    plt.title("単語の出現回数Rank20")
    plt.grid(True)
    plt.subplots_adjust(left=0.112, right=0.95, bottom=0.062, top=0.955)
    plt.yticks(fontsize=7)
    ax = ds[:50].sort_values(ascending=True, axis=0).plot.barh(align='center', color='#1E7F00', linewidth=0)
    plt.savefig(os.path.join('images','barh.png'))
    #plt.show()


def getRequest(url, header, logger):
    retry_count = 0
    max_retry = 10
    # ツイート取得の制限がかかった際に10回リトライする
    while retry_count < max_retry:
        try:
            response = requests.get(url, headers=header, timeout=(5.0, 10.0))
            if response.status_code == requests.codes.ok:
                # ステータスコードが400未満の場合
                break
            elif response.status_code == 429:
                # ツイート取得の制限がかかった際に､制限解除までmax_retry回まで待機
                if retry_count >= max_retry:
                    raise MaxRetryError('リトライ回数の上限に達しました')
                retry_count += 1
                logger.info(f'{retry_count}回目のリトライ')
                time.sleep(retry_count * 10)
            else:
                # ステータスコードが400系や500系だった場合も例外を送出
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.exception("RequestsError:%s", e.response.text)
            sys.exit()
        except MaxRetryError as e:
            logger.exception("MaxRetryError:%s", e)
            sys.exit()
        except requests.exceptions.HTTPError as e:
            logger.exception("HTTPError:%s", e)
            sys.exit()
        except requests.exceptions.ConnectionError as e:
            logger.exception("ConnectionError:%s", e)
            sys.exit()
        except requests.exceptions.Timeout as e:
            logger.exception("TimeoutError:%s", e)
            # print("request failed. response_text=(%s)", e)

    # print(json.dumps(response.json(), indent=4, sort_keys=True))
    return response.json()

def getWordList(data, delimiter, tknzr):
    result = []
    for i, d in enumerate(data):
        # 改行の削除
        d['text'] = re.sub('[\r\n]+', '', d['text'])
        # URLの削除
        d['text'] = re.sub('[ 　\n]?https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', d['text'])
        # ユーザー名の削除
        d['text'] = re.sub('[ 　]?@[a-zA-Z0-9_]+', '', d['text'])
        # 絵文字の除去
        d['text'] = ''.join(filter(lambda c: ord(c) < 0x10000 and ord(c) != 0x2705, d['text']))
        # ハッシュタグの削除
        # data['data'][i]['text'] = re.sub('#.+ ', '', d['text'])
        # 全角スペース、タブ、改行を削除
        d['text'] = re.sub(r'[\u3000\u200d\t\r\n]', '', d['text'])
        # 形態素解析
        tokens = tknzr.tokenize(d['text'])
        result.append(delimiter.join(map(lambda t: t.surface, (filter(lambda t: t.part_of_speech.split(',')[0] in ['名詞','形容詞','動詞'], tokens))))+delimiter)
    return result

def toStrUTC(objDatetime):
    # タイムゾーンの情報を付与（ここではUTC）
    objDatetime = objDatetime.replace(tzinfo=tz(td(hours=+9),'JST'))
    # タイムゾーンを日本時間に変換
    objDatetime = objDatetime.astimezone(tz.utc)
    # タイムゾーン表記を消去（後ろに+09:00（タイムゾーン表記）がつくため）
    objDatetime = objDatetime.replace(tzinfo=None)
    return objDatetime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]+'Z'

def toStrJST(objDatetime):
    # タイムゾーンの情報を付与（ここではUTC）
    objDatetime = objDatetime.replace(tzinfo=tz.utc)
    # タイムゾーンを日本時間に変換
    objDatetime = objDatetime.astimezone(tz(td(hours=+9),'JST'))
    # タイムゾーン表記を消去（後ろに+09:00（タイムゾーン表記）がつくため）
    objDatetime = objDatetime.replace(tzinfo=None)
    return objDatetime.strftime('%Y-%m-%d %H:%M:%S')

class MaxRetryError(Exception):
    pass

if __name__ == "__main__":
    main()

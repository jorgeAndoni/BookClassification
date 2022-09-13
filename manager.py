import pandas as pd
from collections import Counter
import numpy as np
import os
import logging
import random
import embeddings
import network
import classifier
from sklearn.preprocessing import StandardScaler
import csv

def balance_csv(csv_file):
    min_books = 5
    result = csv_file[(csv_file['category'] == 'S') | (csv_file['category'] == 'J')]
    special_cats = ['P', 'D', 'B', 'N', 'H', 'L']
    for i in special_cats:
        pBooks = (csv_file[(csv_file['category'] == i) & (csv_file['words_complete'] >= 8000)])
        random = pBooks.sample(n=min_books)
        result = result.append(random)
    return result



class BookAnalysis(object):

    def __init__(self, logger, csv_file_type ,csv_file, text_partition, limiar, remove_stops, embedding_model, feature_selection, output_folder):
        self.logger = logger
        self.book_csv = csv_file
        self.text_partition_size = text_partition
        self.embedding_percentages = limiar
        self.remove_stops = remove_stops
        self.embedding_model = embedding_model
        self.feature_selection = feature_selection
        self.iterations = 0
        self.output_folder = output_folder
        self.path_results = 'results/' + self.output_folder + '/'
        #self.database_type = 'category' # author
        self.database_type = csv_file_type #'category'  # author

    def get_common_words(self, segments):
        commom_words = segments[0]
        for index, i in enumerate(segments):
            commom_words = list(set(commom_words) & set(i))
        result = {word: index for index, word in enumerate(commom_words)}
        return result

    def get_top_words(self, segments):
        top_words = int(self.feature_selection[self.feature_selection.rfind('_') + 1:])
        all_words = []
        for i in segments:
            all_words.extend(list(set(i)))
        counts = Counter(all_words)
        features = counts.most_common(top_words)
        most_commom = dict()
        for index, feat in enumerate(features):
            most_commom[feat[0]] = index
        return most_commom

    def organize_books(self):
        if self.remove_stops == 'sin_stops':
            key = 'words_filtered'
            key2 = 'filtered_content'
        else: #con_stops
            key = 'words_complete'
            key2 = 'complete_content'
        min_size_filtered = self.text_partition_size
        print('Min book size:', min_size_filtered, '\n')
        self.logger.info('Min book size: ' + str(min_size_filtered) + '\n')
        corpus = list(self.book_csv[key2])
        corpus = [text.split() for text in corpus]
        segmented_corpus = []
        auxiliar_container = []
        size_partitions = []
        for book in corpus:
            partitions = int(round(len(book)/min_size_filtered,2)+0.5)
            segments = [book[int(round(min_size_filtered * i)): int(round(min_size_filtered * (i + 1)))] for i in range(partitions)]
            size_partitions.append(len(segments))
            segmented_corpus.append(segments)
            for i in segments:
                auxiliar_container.append(i)
        if self.feature_selection == 'common_words':
            words_features = self.get_common_words(auxiliar_container)
        else:
            words_features = self.get_top_words(auxiliar_container)
        self.iterations = int(np.mean(size_partitions))
        return corpus, segmented_corpus, words_features



    def analysis(self):
        corpus, corpus_partitions, words_features = self.organize_books()
        classes = list(self.book_csv[self.database_type]) ## or 'author'
        total_classes = list(set(self.book_csv[self.database_type])) ## or author
        number_books = (self.book_csv[self.book_csv[self.database_type] == total_classes[0]]).shape[0]

        text = 'Word features(' + str(len(words_features)) + '): ' + str(words_features)
        print(text)
        self.logger.info(text)


        self.logger.info('Training word embeddings ....')
        objEmb = embeddings.WordEmbeddings(corpus, self.embedding_model)
        model = objEmb.get_embedding_model()
        self.logger.info('Word embeddings sucessfully trained')


        dict_categories = list(set(classes))
        dict_categories = {cat: index for index, cat in enumerate(dict_categories)}

        iteration_scores = []
        #for iteration in range(self.iterations):  #
        for iteration in range(1):
            print('Init of iteration:', iteration + 1)
            self.logger.info('Init of iteration: ' + str(iteration + 1))

            labels = []
            all_features = []

            for index, (book, category) in enumerate(zip(corpus_partitions, classes)):
                random_index = random.randint(0, len(book) - 1)
                print('book:', index + 1)
                print('category:', category)
                print('partitions:', len(book))
                print('iteration: ' + str(iteration + 1) + ' of ' + str(self.iterations))

                #self.logger.info('book: ' + str(index + 1))
                #self.logger.info('category: ' + category)
                #self.logger.info('partitions: ' + str(len(book)))
                #self.logger.info('iteration: ' + str(iteration + 1) + ' of ' + str(self.iterations))
                labels.append(dict_categories[category])
                selected_partition = book[random_index]

                obj = network.CNetwork(selected_partition, model, self.embedding_percentages)
                cNetworks = obj.create_networks()
                #cNetworks = obj.create_filtered_networks()

                features = obj.get_frequency_counts(cNetworks, words_features)
                all_features.append(features)
                print('\n')

            all_features = np.array(all_features)
            print(all_features.shape)
            scaler = StandardScaler(with_mean=True, with_std=True)
            all_scores = []

            for index in range(len(self.embedding_percentages) + 1):
                limiar_features = all_features[:, index]
                limiar_features = scaler.fit_transform(limiar_features)
                obj = classifier.Classification(limiar_features, labels, number_books)
                scores = obj.get_scores()
                all_scores.append(scores)
                print(index, limiar_features.shape, labels)
                print()
            iteration_scores.append(all_scores)
            print('------ End of iteration ' + str(iteration + 1) + ' ------\n\n')


        print('Final results .....')
        iteration_scores = np.array(iteration_scores)
        percs = [str(val) + '%' for val in self.embedding_percentages]
        percs.insert(0, '0%')

        print('Results for texts of ' + str(self.text_partition_size) + ' words')
        final_results = []
        for limiar in range(len(self.embedding_percentages) + 1):
            values = iteration_scores[:, limiar]
            scores = values.mean(axis=0)
            scores = [round(score,2) for score in scores]
            final_results.append(scores)
            print('Final scores:', percs[limiar], scores)

        return final_results


if __name__ == '__main__':

    #books_authors = pd.read_csv('../databases/books_authorship.csv')
    books_authors = pd.read_csv('../databases/books_authorship_english.csv')
    model = 'w2v'
    use_stops = 'con_stops'
    #word_selection = 'top_300' #200 300
    word_selection = 'top_100'  # 200 300
    folder_results = 'test_output'
    #text_size = 1000
    #text_sizes = [1000, 2000, 3000, 4000]
    #text_sizes = [1000, 1500, 2000, 2500, 5000, 10000] # 0
    #text_sizes = [10000, 15000]
    text_sizes = [10000]
    #limiars = [1, 5, 10, 15 , 20, 25, 30, 35,40,45,50,55,60]
    #limiars = [70, 75, 80, 85, 90, 95, 100]
    #limiars = [1, 5, 10, 15, 20, 25, 30, 35,40,45,50]
    limiars = [1, 10,  20, 30,  40,  50, 60, 70, 80, 90, 100]

    path_results = 'results/' + folder_results + '/'
    try:
        os.mkdir(path_results)
    except:
        print("Existe")

    log_file = path_results + 'log_file.log'

    logging.basicConfig(filename=log_file, level=logging.DEBUG)
    logger = logging.getLogger('testing_book_classification')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    #out_file = path_results + 'output.txt'
    #sys.stdout = open(out_file, 'w')

    # 'category'  # author

    db_type = 'author'
    dataset = books_authors

    percs = [str(val) + '%' for val in limiars]
    #percs = []
    #for val in limiars:
    #    if val<10:
    #        percs.append('0' + str(val)+ '%')
    #    else:
    #        percs.append(str(val)+ '%')

    percs.insert(0, '0%')

    all_res = []
    for size in text_sizes:
        print('Working with texts of ' + str(size) + ' words')
        obj = BookAnalysis(logger, csv_file_type=db_type ,csv_file=dataset, text_partition=size, limiar=limiars,
                       remove_stops=use_stops, embedding_model=model, feature_selection=word_selection, output_folder=folder_results)
        scores = obj.analysis()
        all_res.append(scores)



    print('\n\nResultadoss finales')
    header = ['limiar', 'DT', 'KNN', 'NB', 'SVM']
    rows = []
    for index, resultados in enumerate(all_res):
        text_out = 'Results for text of ' + str(text_sizes[index]) + ' words'
        print(text_out)
        rows.append([str(text_sizes[index]) + ' words', '', '', '', ''])
        rows.append(header)
        for aux, limiar in enumerate(resultados):
            text = 'Final scores: ' + str(percs[aux]) + ' --> ' + str(limiar)
            row = limiar
            row.insert(0, str(percs[aux]))
            rows.append(row)
            print(text)

    path = 'testt' +word_selection + '.csv'
    with open(path, mode='w') as myFile:
        writer = csv.writer(myFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            writer.writerow(row)

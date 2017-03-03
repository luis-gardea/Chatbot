#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PA6, CS124, Stanford, Winter 2016
# v.1.0.2
# Original Python code by Ignacio Cases (@cases)
# Ported to Java by Raghav Gupta (@rgupta93) and Jennifer Lu (@jenylu)
######################################################################
import csv
import math
import re

import numpy as np
import string
from movielens import ratings
from random import randint
from PorterStemmer import PorterStemmer

DATA_POINTS = 5
NUM_RECS = 5

epsilon = 1e-4
articles = "a|an|the|la|el|los|las|la|le|l'|les|der|das|da|det|den"
negation_terms = "not|never|n't"
pos_sentiment = ['love','enjoy','like','appreciate', 'recommend']
neg_sentiment = ['hate','dislike']
pos_adjective = ['favorite', 'best','amazing', 'fantastic', 'charming', 'original', 'powerful', 'clever', 'riveting', 'hilarious', 'funny']
neg_adjective = ['worst', 'terrible', 'predictable', 'dreadful', 'confusing', 'disappointing', 'stupid', 'boring']
intensifiers = "really|very|extremely"
emotion_anger = ["angry", "enraged", "mad", "infuriated", "provoked", "livid", "furious", "irate", "aggravated", "outraged", "pissed"]
emotion_happy = ["happy", "excited", "ecstatic", "cheerful", "joy", "euphoric", "joyful", "glad", "elated", "laughing", "turnt"]
emotion_sad = ["sad", "disappointed", "unhappy", "sorry", "dejected", "morose", "depressing", "sorrowful", "distresssed", "distraught"]
exclude = set(string.punctuation)

class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    #############################################################################
    # `moviebot` is the default chatbot. Change it to your chatbot's name       #
    #############################################################################
    def __init__(self, is_turbo=False):
      self.name = 'pablo'
      self.is_turbo = is_turbo

      self.stemmer = PorterStemmer()

      self.read_data()
      self.user_vec = []
      self.rec_list = []
      self.user_cont_flag = True
      self.rec_list_idx = 0;

      self.disambiguate_title_flag = False
      self.disambiguate_list = []
      self.disambiguate_sentiment = ""
      self.disambiguate_input = ""
      self.disambiguated_movie = ""


    #############################################################################
    # 1. WARM UP REPL
    #############################################################################

    def greeting(self):
      """chatbot greeting message"""
      #############################################################################
      # TODO: Write a short greeting message                                      #
      #############################################################################

      greeting_message = "Hi! I\'m Pablo! I\'m going to recommend a movie to you. "
      greeting_message += "First I will ask you about your taste in movies. "
      greeting_message += "Tell me about a movie that you have seen."

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return greeting_message

    def goodbye(self):
      """chatbot goodbye message"""
      #############################################################################
      # TODO: Write a short farewell message                                      #
      #############################################################################

      goodbye_message = 'Thank you for hanging out with me! Stay in touch! Goodbye!'

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return goodbye_message


    #############################################################################
    # 2. Modules 2 and 3: extraction and transformation                         #
    #############################################################################

    def process(self, input):
      """Takes the input string from the REPL and call delegated functions
      that
        1) extract the relevant information and
        2) transform the information into a response to the user
      """
      #############################################################################
      # TODO: Implement the extraction and transformation in this method, possibly#
      # calling other functions. Although modular code is not graded, it is       #
      # highly recommended                                                        #
      #############################################################################
      # if self.is_turbo == True:
      #   print 'processed %s in creative mode!!' % input
      # else:
      #   print 'processed %s in starter mode' % input

      if self.is_turbo and self.disambiguate_title_flag:
        if input.isdigit():
          index = int(input)
          if index >= 0 and index < len(self.disambiguate_list):
            return self.disambiguate_movie(index)
          else:
            return "This value was not in the list. " + self.print_disambiguate_prompt()
        else:
          if input.lower() == "none":
            self.disambiguate_title_flag = False
            self.disambiguate_list = []
            self.disambiguate_sentiment = ""
            self.disambiguated_movie = ""
            self.disambiguate_input = ""
            return "Ok. Tell me about another movie you have seen."
          else:
            return self.print_disambiguate_prompt()

      # If recommendation just given, check answer whether user wants another movie
      # if not self.user_cont_flag:
      #   if re.findall("yes", input.lower()):
      #     return self.print_recommendation()
      #   else:
      #     self.user_cont_flag = True
      #     return "Ok. Tell me about some more movies that you have seen in order for me to provide you with better recommendations."

      movie = re.findall("\"(.+?)\"", input)
      if self.is_turbo and not movie:
        if self.find_movie(input):
          input = self.find_movie(input)
          movie = re.findall("\"(.+?)\"", input)

      sentiment = self.get_sentiment(input)
      if not movie:
        if self.is_turbo:
          emotion = self.handleEmotions(input)
          if emotion:
            return emotion
        if sentiment < self.base_rating: # Guess user doesn't want to continue
          return "I want to hear more about movies! Tell me about another movie you have seen."
        return "Sorry, I don't understand. Tell me about a movie that you have seen."
      if len(movie) > 1:
        return "Please tell me about one movie at a time. Go ahead."

      movie = movie[0]
      if sentiment == self.base_rating:
        return "I'm sorry, I'm not quite sure if you liked \"" + movie + "\". Tell me more about \"" + movie + "\"."

      movie_idx = -1
      movie_idx_list = self.get_movie(movie)
      if len(movie_idx_list) == 0:
        return "I'm sorry, I haven't heard of that movie. Tell me about another movie you have seen."
      elif len(movie_idx_list) == 1:
        movie_idx = movie_idx_list[0]
      else:
        if self.is_turbo:
          self.disambiguate_title_flag = True
          self.disambiguate_list = movie_idx_list
          self.disambiguate_sentiment = sentiment
          self.disambiguated_movie = movie 
          self.disambiguate_input = input
          return self.print_disambiguate_prompt()
        else:
          movie_idx = movie_idx_list[0]

      return self.add_movie(movie_idx, sentiment, input)

    # This will attempt to find a movie when one is listed without quotation marks
    def find_movie(self, input):
      input_sep = input.split()
      results = []
      for i in range(len(input_sep)):
        if input_sep[i].istitle():
          for j in range(i,len(input_sep)+1):
            movie = " ".join(input_sep[i:j])
            if not movie or len(movie.split()) < 1:
              continue
            movie = movie.lower()
            adj_movie = movie.lower()
            if re.findall(articles, movie.split()[0]):
              start_article = movie.split()[0]
              adj_movie = movie.replace(start_article + " ", "")
              adj_movie += ", " + start_article
            for idx,title in enumerate(self.titles):
              if movie in title[0].lower() or adj_movie in title[0].lower():
                if not (movie in results):
                  results.append(movie)
      if len(results) > 0:
        movie = max(results, key=lambda y: len(y))
        input = input.lower().replace(movie,"\"" + movie + "\"")
        return input
      return None


    def add_movie(self, movie_idx, sentiment, input):
      movie = self.titles[movie_idx][0]

      # If movie has already been rated and sentiment is found, then we can safely remove it from the vector
      # and it will be added in with the updated rating, but not increase vector size
      if not self.is_turbo:
        if (movie_idx, 1.0) in self.user_vec:
          self.user_vec.remove( (movie_idx, 1.0) )
        elif (movie_idx, -1.0) in self.user_vec:
          self.user_vec.remove( (movie_idx, -1.0) )

      very = None
      verb = "" # default verb
      adjective = None
      exclamations = 0
      if sentiment >= self.base_rating:
        if self.is_turbo:
          self.user_vec.append( (movie_idx, sentiment) )
        else:
          self.user_vec.append( (movie_idx, 1.0) )
        verb = "like"
        for word in input.split():
          exclamations += len(re.findall("!", word))
          word = word.replace("!","")
          if re.findall(intensifiers,word.lower()):
            very = re.findall(intensifiers, word.lower())[0]
            if very == "very":
              very = "very much"
          for posword in pos_sentiment:
            if self.stemmer.stem(word) == self.stemmer.stem(posword):
              verb = posword
          if word in pos_adjective:
            adjective = word
      elif sentiment < self.base_rating:
        if self.is_turbo:
          self.user_vec.append( (movie_idx, sentiment) )
        else:
          self.user_vec.append( (movie_idx, -1.0) )
        verb = "dislike" #default verb
        for word in input.split():
          exclamations += len(re.findall("!", word))
          word = word.replace("!","")
          if re.findall(intensifiers,word.lower()):
            very = re.findall(intensifiers, word.lower())[0]
            if very == "very":
              very = "very much"
          for negword in neg_sentiment:
            if self.stemmer.stem(word) == self.stemmer.stem(negword):
              verb = negword
          if word == neg_adjective:
            adjective = word

      response = "You"
      if very:
        response += " %s" % very
      response += " %sd \"%s\"" % (verb, movie)
      if adjective:
        response += ", it was the %s" % adjective
      if exclamations > 0:
        response += "!"*exclamations
      else:
        response += "."
      response += " Thank you!"

      if len(self.user_vec) % DATA_POINTS == 0: # Provide new recs every 5 data points
        self.rec_list_idx = 0
        self.rec_list = self.recommend(self.user_vec)
        response += " That's enough for me to make a recommendation. "
        response += "I suggest you watch: "
        for i in range(NUM_RECS):
          response += "\"" + self.print_recommendation() + "\""
          if i != NUM_RECS-1:
            response += ", "
        response += "."
      else:
        response += " Tell me about another movie you have seen."
      return response

    def handleEmotions (self, input):
      movie = re.findall("(\".+?\")", input)
      if movie:
        input = input.replace(movie[0], '')
      input = input.split()
      negation = False
      pos_response = "I'm glad to hear you're happy! Maybe it's time for a new feel good movie! Let me know what movies you like!"
      neg_response = "I'm sorry to hear you're upset. Maybe a movie recommendation will cheer you up. Let me know what movies you like!"
      anger_response = "I'm sorry to hear you're angry. Maybe it's time for a new feel good movie! Let me know what movies you like!"
      not_anger_response = "I'm glad to hear you're not angry. It's a perfect time for a new movie! Let me know what movies you like!"
      for word in input:
        for e in emotion_happy:
          if self.stemmer.stem(word) == self.stemmer.stem(e):
            if not negation:
              return pos_response
            else:
              return neg_response
        for e in emotion_sad:
          if self.stemmer.stem(word) == self.stemmer.stem(e):
            if not negation:
              return neg_response
            else:
              return pos_response
        for e in emotion_anger:
          if self.stemmer.stem(word) == self.stemmer.stem(e):
            if not negation:
              return anger_response
            else:
              return not_anger_response
        if re.findall(negation_terms, word.lower()):
          negation = True
      return None

    def get_sentiment(self, input):
      # First remove movie title from string
      movie = re.findall("(\".+?\")", input)
      if movie:
        input = input.replace(movie[0], '')
      input = input.split()
      # Get pos and neg scores
      # pos = neg = 0.0
      sentiment = 0.0
      negation = False
      intensifier = False
      pos_mult = 1
      neg_mult = 1
      for word in input:
        stemmed_word = ''.join(ch for ch in word if ch not in exclude)
        stemmed_word = self.stemmer.stem(stemmed_word)
        word_sentiment = self.get_word_sentiment(stemmed_word)
        if word_sentiment:
          if intensifier:
            word_sentiment *= 2
            intensifier = False
          if negation:
            sentiment -= word_sentiment
            negation = False
          else:
            sentiment += word_sentiment
        if re.findall(negation_terms, word.lower()):
          negation = True
        if re.findall(intensifiers, word.lower()):
          intensifier = True

      rating = self.base_rating + sentiment/2.0
      if rating > 5.0:
        rating = 5.0
      elif rating < 1.0:
        rating = 1.0
      return self.base_rating + sentiment/2.0

    def get_word_sentiment(self, word):
      if word in self.sentiment:
        if self.sentiment[word] == "pos":
          for pos_word in pos_sentiment:
            if word == self.stemmer.stem(pos_word):
              return 2
          for pos_adj in pos_adjective:
            if word == self.stemmer.stem(pos_adj):
              return 2
          return 1
        else:
          for neg_word in neg_sentiment:
            if word == self.stemmer.stem(neg_word):
              return -2
          for neg_adj in neg_adjective:
            if word == self.stemmer.stem(neg_adj):
              return -2
          return -1
      return None

    def disambiguate_movie(self, index):
      index = self.disambiguate_list[index]
      response = self.add_movie(index, self.disambiguate_sentiment, self.disambiguate_input)
      self.disambiguate_title_flag = False
      self.disambiguate_list = []
      self.disambiguate_sentiment = ""
      self.disambiguated_movie = ""
      return response

    def print_disambiguate_prompt(self):
      response = "We found various movies for \"%s\". Please type in the corresponding number of your intended movie:\n" % self.disambiguated_movie
      for i,movie_idx in enumerate(self.disambiguate_list):
        response += "\t%s: %s\n" % (str(i), self.titles[movie_idx][0])
      response += "Or type \"None\" if you didn't mean any of these."
      return response

    def print_recommendation(self):
      if self.rec_list_idx >= len(self.rec_list):
        self.user_cont_flag = True
        return "I need more information to provide new recommendations. Tell me about another movie you have seen."
      # Print movie recommendation, parse title correctly
      movie = self.titles[self.rec_list[self.rec_list_idx][0]][0]
      remove_vals = re.findall("(\(.*?\))", movie)
      for val in remove_vals:
        movie = movie.replace(val, "").rstrip()
      article_val = re.findall(articles, movie.split()[-1].lower())
      if article_val and re.findall(", " + article_val[-1].capitalize(), movie):
        movie = movie.replace(", " + article_val[-1].capitalize(), "")
        movie = article_val[-1].capitalize() + " " + movie
      # response = "I suggest you watch \"" + movie + "\"."
      response = movie
      self.rec_list_idx += 1
      self.user_cont_flag = False
      return response

    def dameraulevenshtein(self, seq1, seq2):
      """Calculate the Damerau-Levenshtein distance between sequences.

      CITATION: THIS CODE IS FROM PA2 STARTER CODE

      This distance is the number of additions, deletions, substitutions,
      and transpositions needed to transform the first sequence into the
      second. Although generally used with strings, any sequences of
      comparable objects will work.

      Transpositions are exchanges of *consecutive* characters; all other
      operations are self-explanatory.

      This implementation is O(N*M) time and O(M) space, for N and M the
      lengths of the two sequences.

      >>> dameraulevenshtein('ba', 'abc')
      2
      >>> dameraulevenshtein('fee', 'deed')
      2

      It works with arbitrary sequences too:
      >>> dameraulevenshtein('abcd', ['b', 'a', 'c', 'd', 'e'])
      2
      """
      # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
      # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
      # However, only the current and two previous rows are needed at once,
      # so we only store those.
      oneago = None
      thisrow = range(1, len(seq2) + 1) + [0]
      for x in xrange(len(seq1)):
          # Python lists wrap around for negative indices, so put the
          # leftmost column at the *end* of the list. This matches with
          # the zero-indexed strings and saves extra calculation.
          twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
          for y in xrange(len(seq2)):
              delcost = oneago[y] + 1
              addcost = thisrow[y - 1] + 1
              subcost = oneago[y - 1] + (seq1[x] != seq2[y])
              thisrow[y] = min(delcost, addcost, subcost)
              # This block deals with transpositions
              if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                  and seq1[x-1] == seq2[y] and seq1[x] != seq2[y]):
                  thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
      return thisrow[len(seq2) - 1]


    def get_movie(self, movie):
      # Get index of movie in titles array, -1 if absent
      if not movie or len(movie.split()) == 0:
        return -1
      movie = movie.lower()
      year = re.findall("(\([0-9]{4}\))", movie)
      if year:
        movie = movie.replace(year[0], '')
      adj_movie = movie
      if re.findall(articles, movie.split()[0]):
        start_article = movie.split()[0]
        adj_movie = movie.replace(start_article + " ", "")
        adj_movie += ", " + start_article
      results = []
      for idx,title in enumerate(self.titles):
        title = title[0].lower()
        year2 = re.findall("(\([0-9]{4}\))", title)
        if year2 and year:
          if year[0] != year2[0]:
            continue
          # title[0] = title[0].replace(year2[0], '')
        if movie in title or adj_movie in title:
          results.append(idx)

      if self.is_turbo:
        # All of the spell checking code
        if len(results) == 0:
          if movie.split()[-1].isdigit():
            movie = movie[:-1]
            return self.get_movie(movie)

          min_dist = 4
          min_idx = ''
          for idx, title in enumerate(self.titles):
            year2 = re.findall("(\([0-9]{4}\))", title[0])
            title = title[0].lower()
            if year:
              if year != year2:
                continue
            title = title.replace(year2[0], '')

            alternate_title = re.findall("(\([A-Za-z ]+\))",title)
            actual_title = title.strip()

            if alternate_title:
              alternate_title = alternate_title[0].strip()
              actual_title = actual_title.replace(alternate_title,'').strip()

              dist = min(self.dameraulevenshtein(movie, alternate_title),self.dameraulevenshtein(adj_movie, alternate_title))
              if dist <= min_dist:
                if dist < min_dist:
                  results = []
                min_dist = dist 
                min_idx = idx
                results.append(min_idx)

            dist = min(self.dameraulevenshtein(movie, actual_title),self.dameraulevenshtein(adj_movie, actual_title))
            if dist <= min_dist:
              if dist < min_dist:
                results = []
              min_dist = dist
              min_idx = idx
              results.append(min_idx)

      return results


    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    def read_data(self):
      """Reads the ratings matrix from file"""
      # This matrix has the following shape: num_movies x num_users
      # The values stored in each row i and column j is the rating for
      # movie i by user j
      self.titles, self.ratings = ratings()
      self.base_rating = 3.0
      self.binarize()
      reader = csv.reader(open('data/sentiment.txt', 'rb'))
      sentiment = dict(reader)
      self.sentiment = {}
      # added stemming for sentiment keywords
      for key, val in sentiment.items():
        stemmed_key = self.stemmer.stem(key)
        self.sentiment[stemmed_key] = val


    def binarize(self):
      """Modifies the ratings matrix to make all of the ratings binary"""
      # Loop through the ratings and binarize based on overall average rating
      rating_sum = np.sum(self.ratings)
      rating_count = np.count_nonzero(self.ratings)
      rating_avg = (1.0 * rating_sum) / rating_count

      def binary_transform(x, rating_avg):
        if x == 0.0:
          return 0.0
        elif x >= rating_avg:
          return 1.0
        else:
          return -1.0

      btransform = np.vectorize(binary_transform, otypes=[np.float])
      if self.is_turbo:
        self.ratings = btransform(self.ratings, rating_avg)

    def distance(self, u, v):
      """Calculates a given distance function between vectors u and v"""
      # Implement the distance function between vectors u and v]
      # Note: you can also think of this as computing a similarity measure
      # Use of cosine similarity measure, assumes u and v have equal length
      num = np.dot(u,v)
      # den_u = np.sum(u**2)
      # den_v = np.sum(v**2)
      den_u = np.linalg.norm(u)
      den_v = np.linalg.norm(v)
      if den_u == 0.0 or den_v == 0.0:
        return 0.0
      # return num / (math.sqrt(den_u) * math.sqrt(den_v))
      return num / (den_u * den_v)


    def recommend(self, u):
      """Generates a list of movies based on the input vector u using
      collaborative filtering"""
      # Implement a recommendation function that takes a user vector u
      # and outputs a list of movies recommended by the chatbot
      rec_list = []
      for i, movie in enumerate(self.ratings):
        rxi = 0.0
        for tup in u:
          j = tup[0]
          rxj = tup[1]
          if i == j: # Skip movies in user_vec
            continue
          sij = self.distance(self.ratings[i], self.ratings[j])
          rxi += (rxj * sij)
        movie_rank = [i, rxi] # Store movie index and rating
        rec_list.append(movie_rank)
      rec_list = sorted(rec_list, key=lambda x:x[1], reverse = True) 
      return rec_list


    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, input):
      """Returns debug information as a string for the input string from the REPL"""
      # Pass the debug information that you may think is important for your
      # evaluators
      debug_info = 'debug info'
      return debug_info


    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
      return """
      Welcome to our chatbot El Patron, Pablo Gaviria!

      Pablo can utilize binarized or non binarized data for movie recommendations. Use the turbo mode
      for non binarized data with better recommendations or the starter mode to get normal binarized
      data based recommendations. The non binarized dataset works in conjunction with a feature that
      is able to extract a 1 - 5 ranking of input sentiment about a movie to get you better recs!

      Pablo can also handle happy, sad, and angry emotions in turbo mode when a movie title is not
      in the input. E.g. I am angry about the election.
      This feature will only extract emotion if it was unable to find a movie written in the message.

      Our chatbox so far has more fine-grained sentiment extraction, it handles strong sentiment words,
      intensifiers, and exclamation marks. E.g. I really liked "Tarzan", it was my favorite.

      We also have also implemented a feature to disambiguate titles. It gives a list of possible movies
      fitting the input given. The user then selects the correct one or none from the list. 
      E.g. I liked "Harry Potter"

      We then implemented a spell-checker. It uses Damerau-Levenshtein distance. This feature also works
      with the previous feature, as it still attempts to disambiguate between spelling mistakes and also
      between possible results with the same Damerau-Levenshtein distance.
      E.g. I enjoyed "Alice in wndrlnd"

      We have added the functionality to extract foreign and/or alternate titles.

      We also added functionality to extract movie titles without requiring quotation marks. This feature
      also works with the movie disambiguation feature. E.g. I loved Nacho libre
      """


    #############################################################################
    # Auxiliary methods for the chatbot.                                        #
    #                                                                           #
    # DO NOT CHANGE THE CODE BELOW!                                             #
    #                                                                           #
    #############################################################################

    def bot_name(self):
      return self.name


if __name__ == '__main__':
    Chatbot()

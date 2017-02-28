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

epsilon = 1e-4
articles = "a|an|the|la|el"
negation_terms = "not|never|n't"
strong_sentiment = ['love','hate','favorite','amazing','terrible','worst']
intensifiers = "very|extremely|really|too|totally|super"
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
      for i,word in enumerate(strong_sentiment):
        strong_sentiment[i] = self.stemmer.stem(word)
      self.read_data()
      self.user_vec = []
      self.rec_list = []
      self.user_cont_flag = True
      self.rec_list_idx = 0;
      

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
    #   if self.is_turbo == True:
    #     response = 'processed %s in creative mode!!' % input
    #   else:
    #     response = 'processed %s in starter mode' % input

      # If recommendation just given, check answer whether user wants another movie
      # return self.get_movie(input)

      if not self.user_cont_flag:
        if re.findall("yes", input.lower()):
          return self.print_recommendation()
        else:
          self.user_cont_flag = True
          return "Ok. Tell me about some more movies that you have seen in order for me to provide you with better recommendations."

      movie = re.findall("\"(.+?)\"", input)
      sentiment = self.get_sentiment(input)
      if not movie:
        if sentiment == "neg": # Guess user doesn't want to continue
          return "I want to hear more about movies! Tell me about another movie you have seen."
        return "Sorry, I don't understand. Tell me about a movie that you have seen."
      if len(movie) > 1:
        return "Please tell me about one movie at a time. Go ahead."

      response = ""
      movie = movie[0]
      movie_idx = self.get_movie(movie)
      if movie_idx == -1:
        return "I'm sorry, I haven't heard of that movie. Tell me about another movie you have seen."
      if not sentiment:
        return "I'm sorry, I'm not quite sure if you liked \"" + movie + "\". Tell me more about \"" + movie + "\"."

      # If movie has already been rated and sentiment is found, then we can safely remove it from the vector
      # and it will be added in with the updated rating, but not increase vector size
      if (movie_idx, 1.0) in self.user_vec:
        self.user_vec.remove( (movie_idx, 1.0) )
      elif (movie_idx, -1.0) in self.user_vec:
        self.user_vec.remove( (movie_idx, -1.0) )
        

      if sentiment == "pos":
        self.user_vec.append( (movie_idx, 1.0) )
        response = "You liked \"" + movie + "\". Thank you!"
      else:
        self.user_vec.append( (movie_idx, -1.0) )
        response = "You did not like \"" + movie + "\". Thank you!"

      if len(self.user_vec) % DATA_POINTS == 0: # Provide new recs every 5 data points
        self.rec_list_idx = 0
        self.rec_list = self.recommend(self.user_vec)
        response += " That's enough for me to make a recommendation. "
        response += self.print_recommendation()
      else:
        response += " Tell me about another movie you have seen."
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
      response = "I suggest you watch \"" + movie + "\". Would you like to hear another recommendation? (Or enter :quit if you're done.)"
      self.rec_list_idx += 1
      self.user_cont_flag = False
      return response


    def get_movie(self, movie):
      # Get index of movie in titles array, -1 if absent
      if not movie or len(movie.split()) == 0:
        return -1
      adj_movie = movie
      if re.findall(articles, movie.split()[0].lower()):
        start_article = movie.split()[0]
        adj_movie = movie.replace(start_article + " ", "")
        adj_movie += ", " + start_article
      for idx,title in enumerate(self.titles):
        if movie in title[0] or adj_movie in title[0]:
          return idx
      return -1


    def get_sentiment(self, input):
      # First remove movie title from string
      movie = re.findall("(\".+?\")", input)
      if movie:
        input = input.replace(movie[0], '')
      input = input.split()
      # Get pos and neg scores
      # pos = neg = 0.0
      sentiment = 0
      negation = False
      intense = False
      for word in input:
        word_sentiment = self.get_word_sentiment(word)
        print word, word_sentiment
        if word_sentiment:
          if intense:
            word_sentiment *= 2
            intense = False
          if negation:
            sentiment -= word_sentiment
            negation = False
          else:
            sentiment += word_sentiment

        if re.findall(negation_terms, word.lower()):
          negation = True
        if re.findall(intensifiers, word.lower()):
          insense = True
      # Return input's overall sentiment
      if sentiment > epsilon:
        return "pos"
      elif sentiment < -epsilon:
        return "neg"
      else:
        return None


    def get_word_sentiment(self, word):
      # Get sentiment for word, checking for word variants and negatives
      coeff = 1
      if len(re.findall('!', word)) > 1:
        coeff *= 2
      stemmed_word = ''.join(ch for ch in word if ch not in exclude)

      stemmed_word = self.stemmer.stem(stemmed_word)
      if stemmed_word in strong_sentiment:
        coeff *= 2

      if stemmed_word in self.sentiment:
        if self.sentiment[stemmed_word] == "pos":
          return 1*coeff
        else:
          return -1*coeff
      # if word[-1] == "d" and word[:-1] in self.sentiment:
      #   return self.sentiment[word[:-1]]
      # if word[-2:] == "ed" and word[:-2] in self.sentiment:
      #   return self.sentiment[word[:-2]]
      return None


    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    def read_data(self):
      """Reads the ratings matrix from file"""
      # This matrix has the following shape: num_movies x num_users
      # The values stored in each row i and column j is the rating for
      # movie i by user j
      self.titles, self.ratings = ratings()
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
      self.ratings = btransform(self.ratings, rating_avg)

    def distance(self, u, v):
      """Calculates a given distance function between vectors u and v"""
      # Implement the distance function between vectors u and v]
      # Note: you can also think of this as computing a similarity measure
      # Use of cosine similarity measure, assumes u and v have equal length
      num = np.dot(u,v)
      den_u = np.sum(u**2)
      den_v = np.sum(v**2)
      if den_u == 0.0 or den_v == 0.0:
        return 0.0
      return num / (math.sqrt(den_u) * math.sqrt(den_v))


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
      Your task is to implement the chatbot as detailed in the PA6 instructions.
      Remember: in the starter mode, movie names will come in quotation marks and
      expressions of sentiment will be simple!
      Write here the description for your own chatbot!

      Our chatbox so far has more fine-grained sentiment extraction, it handles strong sentiment words,
      intensifiers, and exclamation marks.
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

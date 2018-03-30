import re
import time

max_edit_distance = 3
verbose = 0
# 0: top suggestion
# 1: all suggestions of smallest edit distance
# 2: all suggestions <= max_edit_distance (slower, no early termination)

longest_word_length = 0

def get_deletes_list(w):
	'''given a word, derive strings with up to max_edit_distance characters
	   deleted'''
	deletes = []
	queue = [w]
	for d in range(max_edit_distance):
		temp_queue = []
		for word in queue:
			if len(word)>1:
				for c in range(len(word)):  # character index
					word_minus_c = word[:c] + word[c+1:]
					if word_minus_c not in deletes:
						deletes.append(word_minus_c)
					if word_minus_c not in temp_queue:
						temp_queue.append(word_minus_c)
		queue = temp_queue

	return deletes

def create_dictionary_entry(dictionary, w):
	'''add word and its derived deletions to dictionary'''
	# check if word is already in dictionary
	# dictionary entries are in the form: (list of suggested corrections,
	# frequency of word in corpus)
	global longest_word_length
	new_real_word_added = False
	if w in dictionary:
		# increment count of word in corpus
		dictionary[w] = (dictionary[w][0], dictionary[w][1] + 1)
	else:
		dictionary[w] = ([], 1)
		longest_word_length = max(longest_word_length, len(w))

	if dictionary[w][1]==1:
		# first appearance of word in corpus
		# n.b. word may already be in dictionary as a derived word
		# (deleting character from a real word)
		# but counter of frequency of word in corpus is not incremented
		# in those cases)
		new_real_word_added = True
		deletes = get_deletes_list(w)
		for item in deletes:
			if item in dictionary:
				# add (correct) word to delete's suggested correction list
				dictionary[item][0].append(w)
			else:
				# note frequency of word in corpus is not incremented
				dictionary[item] = ([w], 0)

	return new_real_word_added

def create_dictionary(dictionary, fname):

	total_word_count = 0
	unique_word_count = 0

	with open(fname) as file:
		print "Creating dictionary..."
		for line in file:
			# separate by words by non-alphabetical characters
			words = re.findall('[a-z]+', line.lower())
			for word in words:
				total_word_count += 1
				if create_dictionary_entry(dictionary, word):
					unique_word_count += 1

	return dictionary

def dameraulevenshtein(seq1, seq2):
	"""Calculate the Damerau-Levenshtein distance between sequences.
	This method has not been modified from the original.
	Source: http://mwh.geek.nz/2009/04/26/python-damerau-levenshtein-distance/

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

def get_suggestions(dictionary, string, silent=False):
	'''return list of suggested corrections for potentially incorrectly
	   spelled word'''
	if (len(string) - longest_word_length) > max_edit_distance:
		if not silent:
			print "no items in dictionary within maximum edit distance"
		return []

	global verbose
	suggest_dict = {}
	min_suggest_len = float('inf')

	queue = [string]
	q_dictionary = {}  # items other than string that we've checked

	while len(queue)>0:
		q_item = queue[0]  # pop
		queue = queue[1:]

		# early exit
		if ((verbose<2) and (len(suggest_dict)>0) and
			  ((len(string)-len(q_item))>min_suggest_len)):
			break

		# process queue item
		if (q_item in dictionary) and (q_item not in suggest_dict):
			if (dictionary[q_item][1]>0):
			# word is in dictionary, and is a word from the corpus, and
			# not already in suggestion list so add to suggestion
			# dictionary, indexed by the word with value (frequency in
			# corpus, edit distance)
			# note q_items that are not the input string are shorter
			# than input string since only deletes are added (unless
			# manual dictionary corrections are added)
				assert len(string)>=len(q_item)
				suggest_dict[q_item] = (dictionary[q_item][1],
										len(string) - len(q_item))
				# early exit
				if ((verbose<2) and (len(string)==len(q_item))):
					break
				elif (len(string) - len(q_item)) < min_suggest_len:
					min_suggest_len = len(string) - len(q_item)

			# the suggested corrections for q_item as stored in
			# dictionary (whether or not q_item itself is a valid word
			# or merely a delete) can be valid corrections
			for sc_item in dictionary[q_item][0]:
				if (sc_item not in suggest_dict):

					# compute edit distance
					# suggested items should always be longer
					# (unless manual corrections are added)
					assert len(sc_item)>len(q_item)

					# q_items that are not input should be shorter
					# than original string
					# (unless manual corrections added)
					assert len(q_item)<=len(string)

					if len(q_item)==len(string):
						assert q_item==string
						item_dist = len(sc_item) - len(q_item)

					# item in suggestions list should not be the same as
					# the string itself
					assert sc_item!=string

					# calculate edit distance using, for example,
					# Damerau-Levenshtein distance
					item_dist = dameraulevenshtein(sc_item, string)

					# do not add words with greater edit distance if
					# verbose setting not on
					if ((verbose<2) and (item_dist>min_suggest_len)):
						pass
					elif item_dist<=max_edit_distance:
						assert sc_item in dictionary  # should already be in dictionary if in suggestion list
						suggest_dict[sc_item] = (dictionary[sc_item][1], item_dist)
						if item_dist < min_suggest_len:
							min_suggest_len = item_dist

					# depending on order words are processed, some words
					# with different edit distances may be entered into
					# suggestions; trim suggestion dictionary if verbose
					# setting not on
					if verbose<2:
						suggest_dict = {k:v for k, v in suggest_dict.items() if v[1]<=min_suggest_len}

		# now generate deletes (e.g. a substring of string or of a delete)
		# from the queue item
		# as additional items to check -- add to end of queue
		assert len(string)>=len(q_item)

		# do not add words with greater edit distance if verbose setting
		# is not on
		if ((verbose<2) and ((len(string)-len(q_item))>min_suggest_len)):
			pass
		elif (len(string)-len(q_item))<max_edit_distance and len(q_item)>1:
			for c in range(len(q_item)): # character index
				word_minus_c = q_item[:c] + q_item[c+1:]
				if word_minus_c not in q_dictionary:
					queue.append(word_minus_c)
					q_dictionary[word_minus_c] = None  # arbitrary value, just to identify we checked this

	# queue is now empty: convert suggestions in dictionary to
	# list for output
	if not silent and verbose!=0:
		print "number of possible corrections: %i" %len(suggest_dict)
		print "  edit distance for deletions: %i" % max_edit_distance

	# output option 1
	# sort results by ascending order of edit distance and descending
	# order of frequency
	#	 and return list of suggested word corrections only:
	# return sorted(suggest_dict, key = lambda x:
	#			   (suggest_dict[x][1], -suggest_dict[x][0]))

	# output option 2
	# return list of suggestions with (correction,
	#								  (frequency in corpus, edit distance)):
	as_list = suggest_dict.items()
	outlist = sorted(as_list, key=lambda(term, (freq, dist)): (dist, -freq))

	if verbose==0:
		if len(outlist)==0:
			return []
		return outlist[0]
	else:
		return outlist

	'''
	Option 1:
	['file', 'five', 'fire', 'fine', ...]

	Option 2:
	[('file', (5, 0)),
	 ('five', (67, 1)),
	 ('fire', (54, 1)),
	 ('fine', (17, 1))...]
	'''

def best_word(dictionary, s, silent=False):
	try:
		return get_suggestions(dictionary, s, silent)[0]
	except:
		return s

def correct_document(dictionary, fname, outfname, printlist=True):
	# correct an entire document
	with open(outfname,'w') as outfile:
		with open(fname) as file:
			doc_word_count = 0
			corrected_word_count = 0
			unknown_word_count = 0

			for i, line in enumerate(file):
				doc_words = re.findall("[a-z]+|[^a-z]+",line.lower())
				for doc_word in doc_words:
					doc_word_count += 1
					if doc_word[0].isalpha():
						suggestion = get_suggestions(dictionary, doc_word)
						if len(suggestion)==0:
							outfile.write(doc_word)
							unknown_word_count += 1
						else:
							outfile.write(suggestion[0])
							if suggestion[0]!=doc_word:
								corrected_word_count += 1
					else:
						outfile.write(doc_word)

## main
def spellcheck(dictionary, word_in):
	correct_document(dictionary, word_in, "output.txt")

if __name__ == "__main__":
	print "Please wait..."
	time.sleep(2)
	start_time = time.time()
	dictionary = {}
	create_dictionary(dictionary,"./total_medicine_data.txt")
	run_time = time.time() - start_time
	print '-----'
	print '%.2f seconds to run' % run_time
	print '-----'

	print " "
	print "Word correction"
	print "---------------"


	while True:
		word_in = raw_input('Enter your file path (or enter to exit): ')
		if len(word_in)==0:
			print "goodbye"
			break
		start_time = time.time()
		print correct_document(dictionary, word_in, "output.txt")
		run_time = time.time() - start_time
		print '-----'
		print '%.5f seconds to run' % run_time
		print '-----'
	

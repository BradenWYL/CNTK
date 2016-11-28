import numpy as np

# determines if print intermediate results
printLog = True

class Node:
	def __init__(self, ID, phrase, pos, ifHighlighted):
		self.phrase = phrase # content of the Node
		self.dep_min = 0.0	# the minimum dependency member
		self.dep_max = 0.0	# the maximum dependency member
		self.pos = pos # part-of-speech
		self.prob_min = 0.0 # the minimum probability member, = frequency for single words
		self.prob_max = 0.0 # the maximum probability member, = frequency for single words
		self.ifHighlighted = ifHighlighted # 1 for positive, 0 for negative, -1 for unknown
		self.child = [] # the list of children, from last to first
		self.length = len(phrase.split()) # length of words in this Node
		self.ID = ID # the ID of this Node
		self.wordID = -1 # -1 for all non-leaf Nodes
		self.order = -1 # the wordID of first word in this Node
		self.feature = np.zeros((1, 58))
		self.label = -1 # seems like the same with 'ifHighlighted'

	def set_child(self, childs):
		for i in childs:
			self.child.append(i)

	def set_prob_min(self, prob):
		self.prob_min = prob

	def set_prob_max(self, prob):
		self.prob_max = prob

	def set_dep_min(self, dep):
		self.dep_min = dep

	def set_dep_max(self, dep):
		self.dep_max = dep

	def __str__ (self):
		s = ""
		s = s + str(self.ID)
		s = s + " " + str(self.pos)
		s = s + " " + self.phrase
		s = s + " " + str(self.wordID)
		s = s + " " + str(self.child)
		s = s + " " + str(self.dep_min)
		s = s + " " + str(self.dep_max)
		s = s + " " + "{:.3f}".format(self.prob_min)
		s = s + " " + "{:.3f}".format(self.prob_max)
		s = s + " " + str(self.ifHighlighted)
		return s

	def vector(self):
		self.feature[0, self.pos] = 1;
		self.feature[0, 54] = self.dep_min;
		self.feature[0, 55] = self.dep_max;
		self.feature[0, 56] = self.prob_min;
		self.feature[0, 57] = self.prob_max;
		self.label = self.ifHighlighted
		return self.feature

def parseToTree(lines):
	# This function reads the information in the Stanford Parser output and reconstruct a tree
	Nodes_by_id = {}
	stack_pos = [] #part-of-speech
	stack_node = [] #IDs
	stack_childs = [] #IDs
	node_id = 1
	wordID = 1
	data = ' '.join([line.replace('\n', '') for line in lines]) # re-organize input into a well separated string
	tokenized = data.split() # tokenize the input into words
	for token in tokenized:
		if token.startswith('('):
			stack_pos.append(token[1:])
			stack_node.append(node_id)
			node_id += 1
		else:
			l = len(token)
			cut = token.find(')')
			word = token[:cut]
			stack_childs.append(stack_node.pop()) #Pop ID from node and push to childs
			newNode = Node(stack_childs[-1], word, stack_pos.pop(), '-1') #create the Node
			newNode.wordID = wordID
			newNode.order = wordID
			wordID += 1
			Nodes_by_id[newNode.ID] = newNode
			for j in range(1,l-cut):
				node_id_tmp = stack_node.pop()
				pos_tmp = stack_pos.pop()
				childs = []
				phrase = ''
				while (stack_childs and stack_childs[-1] > node_id_tmp):
					childs.append(stack_childs.pop())
					if phrase is '':
						phrase = Nodes_by_id[childs[-1]].phrase + phrase
					else:
						phrase = Nodes_by_id[childs[-1]].phrase + ' ' + phrase
				newNode = Node(node_id_tmp, phrase, pos_tmp, '-1')
				newNode.set_child(childs)
				newNode.order = Nodes_by_id[childs[-1]].order
				Nodes_by_id[node_id_tmp] = newNode
				stack_childs.append(node_id_tmp)
	return Nodes_by_id

def parseToDep (deps):
	# This is a word counter, putting each word appearing into the dictionary, value = number of appearence
	dic = {}
	for line in deps:
		start = line.find ("-") + 1
		end = line.find (",", start)
		while ord (line[start]) < 48 or ord (line[start]) > 57:
			start = line.find ("-", start) + 1
		word = int(line [start:end])
		count = dic.get (word, 0)
		dic [word] = count + 1

	return dic

def matchHighlight (TreeTest, sentence):
	# This is the function that reads the highlighted input file and label each Node with '1' or '0'
	sentence = sentence.lstrip()
	if printLog:
		print(sentence)
	period = False
	start = False
	hightlightDict = {}
	wordCount = 1
	for i in sentence:
		if i is '#':
			if start is False:
				start = True
				tmp = wordCount
			else:
				start = False
				hightlightDict[tmp] = wordCount - tmp + 1
		elif i is ' ' or (not i.isalpha() and not i.isdigit()):
			wordCount += 1

	if printLog:
		print(hightlightDict)

	for j in TreeTest:
		if TreeTest[j].order in hightlightDict and TreeTest[j].length == hightlightDict[TreeTest[j].order]:
			TreeTest[j].ifHighlighted = 1
		else:
			TreeTest[j].ifHighlighted = 0
	return TreeTest

def fillDetails (TreeTest, DepTest):
	for i in reversed(range (len (TreeTest))):
		node = TreeTest[i + 1]

		if node.ifHighlighted == 1:
			words = node.phrase.split (" ")
			for w in words:
				bank.add (w)

		if node.wordID != -1:
			prob = bank.count(node.phrase)
			node.set_prob_min (prob)
			node.set_prob_max (prob)

			dep = DepTest.get (node.wordID, 0)
			node.set_dep_min (dep)
			node.set_dep_max (dep)
		else:
			dep_max = []
			dep_min = []
			prob_max = []
			prob_min = []

			for c in node.child:
				dep_max.append (TreeTest [c].dep_max)
				dep_min.append (TreeTest [c].dep_min)
				prob_max.append (TreeTest [c].prob_max)
				prob_min.append (TreeTest [c].prob_min)

			node.set_prob_max (max (prob_max))
			node.set_prob_min (min (prob_min))

			node.set_dep_max (max (dep_max))
			node.set_dep_min (min (dep_min))

		node.pos = pos_convert (node.pos)

	return TreeTest

def handleFile (filename1, filename2):
	file = open (filename1, "r")
	tree = []
	deps = []
	tree_input = True
	blank = False
	DepTest = {}
	TreeTest = {}
	returnValue = []

	file2 = open(filename2, "r")
	data_file2 = ''.join([line.replace('\n', ' ') for line in file2.readlines()])

	for line in file:
		if len (line) <= 1:
			if tree_input and not blank:
				tree_input = False
				blank = True
				continue

			if not tree_input and not blank:
				tree_input = True
				blank = True
				continue
			continue

			continue
		if tree_input:
			if blank:
				blank = False

				DepTest = parseToDep (deps)
				if printLog:
					for key, value in DepTest.items():
						print (str(key) + " " +  str(value))
					print ("------")
				deps = []
				#-------------------------------#
				returnValue.append (fillDetails (TreeTest, DepTest))

			tree.append (line)
		else:
			if blank:
				blank = False

				TreeTest = parseToTree (tree)
				cut1 = data_file2.find('.')
				cut2 = data_file2.find('!')
				cut3 = data_file2.find('?')
				cut = max(cut1, cut2, cut3)
				if cut1 is not -1:
					cut = min(cut, cut1)
				if cut2 is not -1:
					cut = min(cut, cut2)
				if cut3 is not -1:
					cut = min(cut, cut3)
				cut +=1
				if printLog:
					print(cut1)
					print(cut2)
					print(cut3)
					print(cut)
				if cut < len(data_file2) and data_file2[cut] == '#':
					cut += 1
				sentence = data_file2[:cut]
				data_file2 = data_file2[cut:]
				TreeTest = matchHighlight (TreeTest, sentence)
				if printLog:
					for i in TreeTest:
						print (TreeTest[i])
						print (TreeTest[i].order)
						print (TreeTest[i].length)
					print ("------")
				tree = []

			deps.append (line)

	if len (tree) != 0:
		TreeTest = parseToTree (tree)
		cut1 = data_file2.find('.')
		cut2 = data_file2.find('!')
		cut3 = data_file2.find('?')
		cut = max(cut1, cut2, cut3)
		if cut1 is not -1:
			cut = min(cut, cut1)
		if cut2 is not -1:
			cut = min(cut, cut2)
		if cut3 is not -1:
			cut = min(cut, cut3)
		cut += 1
		if cut < len(data_file2) and data_file2[cut] == '#':
			cut += 1
		sentence = data_file2[:cut]
		data_file2 = data_file2[cut:]
		TreeTest = matchHighlight (TreeTest, sentence)
		if printLog:
			for i in TreeTest:
				print (TreeTest[i])
				print (TreeTest[i].order)
				print (TreeTest[i].length)
		tree = []

	if len (deps) != 0:
		DepTest = parseToDep (deps)
		if printLog:
			for key, value in DepTest.items():
				print (str(key) + " " +  str(value))
		deps = []
		#-------------------------------#
		returnValue.append (fillDetails (TreeTest, DepTest))

	return returnValue

class Bank:

	def __init__ (self):
		self.dictionary = {}
		self.total = 0
		return

	def add (self, word):
		w = word.lower ()
		count = self.dictionary.get (w, 0)
		self.dictionary [w] = count + 1
		self.total += 1
		return

	def count (self, word):
		w = word.lower ()
		if self.total == 0:
			return 0
		return self.dictionary.get (w, 0) / self.total

def pos_convert (pos):
	if pos == "CC":		# Coordinating conjunction
		return 1
	if pos == "CD":		# Cardinal number
		return 2
	if pos == "DT":		# Determiner
		return 3
	if pos == "EX":		# Existential there
		return 4
	if pos == "FW":		# Foreign word
		return 5
	if pos == "IN":		# Preposition or subordinating conjunction
		return 6
	if pos == "JJ":		# Adjective
		return 7
	if pos == "JJR":	# Adjective, comparative
		return 8
	if pos == "JJS":	# Adjective, superlative
		return 9
	if pos == "LS":		# List iterm marker
		return 10
	if pos == "MD":		# Modal
		return 11
	if pos == "NN":		# Noun, singular or mass
		return 12
	if pos == "NNS":	# Noun, plural
		return 13
	if pos == "NNP":	# Proper noun, singular
		return 14
	if pos == "NNPS":	# Proper noun, plural
		return 15
	if pos == "PDT":	# Predeterminer
		return 16
	if pos == "POS":	# Possessive ending
		return 17
	if pos == "PRP":	# Personal pronoun
		return 18
	if pos == "PRP$":	# Possessive pronoun
		return 19
	if pos == "RB":		# Adverb
		return 20
	if pos == "RBR":	# Adverb, comparative
		return 21
	if pos == "RBS":	# Adverb, superlative
		return 22
	if pos == "RP":		# Particle
		return 23
	if pos == "SYM":	# Symbol
		return 24
	if pos == "TO":		# to
		return 25
	if pos == "UH":		# Interjection
		return 26
	if pos == "VB":		# Verb, base form
		return 27
	if pos == "VBD":	# Verb, past tense
		return 28
	if pos == "VBG":	# Verb, gerund or present participle
		return 29
	if pos == "VBN":	# Verb, past participle
		return 30
	if pos == "VBP":	# Verb, non-3rd person singular present
		return 31
	if pos == "VBZ":	# Verb, 3rd person singular present
		return 32
	if pos == "WDT":	# Wh-determinder
		return 33
	if pos == "WP":		# Wh-pronoun
		return 34
	if pos == "WP$":	# Possessive wh-pronoun
		return 35
	if pos == "WRB":	# Wh-adverb
		return 36
						# ------- PHRASES? ------#

	if pos == "ADJP":	# Adjective phrase
		return 37
	if pos == "ADVP":	# Adverb phrase
		return 38
	if pos == "NP":		# Noun phrase
		return 39
	if pos == "PP":		# Prepositional phrase
		return 40
	if pos == "S":		# Simple declarative clause
		return 41
	if pos == "SBAR":	# Subordinate clause
		return 42
	if pos == "SBARQ":	# Direct question introduced by wh-element
		return 43
	if pos == "SINV":	# Declarative sentence with subject-aux inversion
		return 44
	if pos == "SQ":		# Yes/no question s and subconstituent of SBARQ excluding wh-element
		return 45
	if pos == "VP":		# Verb phrase
		return 46
	if pos == "WHADVP":	# Wh-adverb phrase
		return 47
	if pos == "WHNP":	# Wh-noun phrase
		return 48
	if pos == "WHPP":	# Wh-prepositional phrase
		return 49
	if pos == "X":		# Constituent of unknown or uncertain category
		return 50
	if pos == "*":		# "Understood" subject of infinitive or imperative
		return 51
	if pos == "0":		# Zero variant of that in subordinate clauses
		return 52
	if pos == "T":		# Trace of wh-Constituent
		return 53

	return 0

bank = Bank ()
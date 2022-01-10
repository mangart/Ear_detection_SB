import math
import numpy as np

class Evaluation:

	def compute_rank1(self, Y, y):
		#for i in range(0,len(Y)):
			#Y[i][i] = math.inf
		classes = np.unique(sorted(y))
		count_all = 0
		count_correct = 0
		for cla1 in classes:
			idx1 = y==cla1
			if (list(idx1).count(True)) <= 1:
				continue
			# Compute only for cases where there is more than one sample:
			Y1 = Y[idx1==True, :]
			Y1[Y1==0] = math.inf
			for y1 in Y1:
				s = np.argsort(y1)
				smin = s[0]
				imin = idx1[smin]
				count_all += 1
				if imin:
					count_correct += 1
		return count_correct/count_all*100


	def compute_rankc(self, Y, y,c=5):
		# Y vsebuje matriko 250x250 vseh razdalj vsake od slik od vsake druge vrednosti na diagonali so razdalje slike od same  sebe in so, zato postavljene na 
		# neskončno
		# y je tabela, ki vsebuje numerične vrednosti dejanskih razredov h katerim pripadajo slike to je tabela 250 vrednosti, ki predstavljajo razred slike
		# nastavimo vrednosti diagonalnih elementov na neskončno
		for i in range(0,len(Y)):
			Y[i][i] = math.inf
		# pridobimo unikatne razrede urejene po velikosti
		classes = np.unique(sorted(y))
		razredi_po_slikah = []
		# ustvarimo števec vseh preverjanj in vseh pravilno razrščenih slik
		count_all = 0
		count_correct = 0
		for i in range(len(Y)):
			classes1 = []
			for j in range(101):
				classes1.append(math.inf)
			# sprehodimo se čez vse razrede
			for cla1 in classes:
				#idx1 vsebuje vrednosti True in False True tam kjer ima slika isti razred kot cla1
				idx1 = y==cla1
				# če k razredu pripada samo ena slika jo preskočimo, saj jo ne moremo pravilno klasificirati, ker bi bila njena vrednost neskončno in s tem bi bila
				# napačno klasificirana, kar pa bi bilo narobe
				if (list(idx1).count(True)) <= 1:
					continue
				# Compute only for cases where there is more than one sample:
				# v Y1 shranimo vse razdalje za slike, ki imajo isti razred
				Y2 = Y[i,idx1==True]
				classes1[cla1] = min(Y2)
			razredi_po_slikah.append(classes1)
		for i in range(len(razredi_po_slikah)):
			razredi = np.argsort(razredi_po_slikah[i])
			if c > len(razredi):
				c = len(razredi)
			clas = y[i]
			counter = 0
			for m in range(0,len(y)):
				if y[i] == y[m]:
					counter += 1
			if counter <= 1:
				continue
			for j in range(0,c):
				if razredi[j] == y[i]:
					count_correct += 1
					break
			count_all += 1
		return count_correct/count_all*100


	def CMC_plot(self,Y,y,ime_grafa):
		vsi_ranki = []
		razredi_po_slikah = self.rankc_for_plot(Y,y)
		for i in range(1,101):
			rank = self.calc_rankc(razredi_po_slikah,y,i)
			vsi_ranki.append(rank)
		import matplotlib.pyplot as plt
		y = vsi_ranki
		x = []
		for i in range(1,101):
			x.append(i)
		plt.figure(figsize=(20, 15))
		plt.title(ime_grafa)
		plt.xlabel("rank")
		plt.ylabel("procent ujemanja") 
		plt.plot(x, y,linewidth = 4) 
		plt.ylim(ymin=0)
		plt.savefig(ime_grafa)

	def calc_rankc(self,razredi_po_slikah,y,c):
		count_all = 0
		count_correct = 0
		for i in range(len(razredi_po_slikah)):
			razredi = np.argsort(razredi_po_slikah[i])
			if c > len(razredi):
				c = len(razredi)
			clas = y[i]
			counter = 0
			for m in range(0,len(y)):
				if y[i] == y[m]:
					counter += 1
			if counter <= 1:
				continue
			for j in range(0,c):
				if razredi[j] == y[i]:
					count_correct += 1
					break
			count_all += 1
		return count_correct/count_all*100           

	def rankc_for_plot(self, Y, y):
		# Y vsebuje matriko 250x250 vseh razdalj vsake od slik od vsake druge vrednosti na diagonali so razdalje slike od same  sebe in so, zato postavljene na 
		# neskončno
		# y je tabela, ki vsebuje numerične vrednosti dejanskih razredov h katerim pripadajo slike to je tabela 250 vrednosti, ki predstavljajo razred slike
		# nastavimo vrednosti diagonalnih elementov na neskončno
		for i in range(0,len(Y)):
			Y[i][i] = math.inf
		# pridobimo unikatne razrede urejene po velikosti
		classes = np.unique(sorted(y))
		razredi_po_slikah = []
		# ustvarimo števec vseh preverjanj in vseh pravilno razrščenih slik
		count_all = 0
		count_correct = 0
		for i in range(len(Y)):
			classes1 = []
			for j in range(101):
				classes1.append(math.inf)
			# sprehodimo se čez vse razrede
			for cla1 in classes:
				#idx1 vsebuje vrednosti True in False True tam kjer ima slika isti razred kot cla1
				idx1 = y==cla1
				# če k razredu pripada samo ena slika jo preskočimo, saj jo ne moremo pravilno klasificirati, ker bi bila njena vrednost neskončno in s tem bi bila
				# napačno klasificirana, kar pa bi bilo narobe
				if (list(idx1).count(True)) <= 1:
					continue
				# Compute only for cases where there is more than one sample:
				# v Y1 shranimo vse razdalje za slike, ki imajo isti razred
				Y2 = Y[i,idx1==True]
				classes1[cla1] = min(Y2)
			razredi_po_slikah.append(classes1)
		return razredi_po_slikah
        

	# Add your own metrics here, such as rank5, (all ranks), CMC plot, ROC, ...

		# def compute_rank5(self, Y, y):
	# 	# First loop over classes in order to select the closest for each class.
	# 	classes = np.unique(sorted(y))
		
	# 	sentinel = 0
	# 	for cla1 in classes:
	# 		idx1 = y==cla1
	# 		if (list(idx1).count(True)) <= 1:
	# 			continue
	# 		Y1 = Y[idx1==True, :]

	# 		for cla2 in classes:
	# 			# Select the closest that is higher than zero:
	# 			idx2 = y==cla2
	# 			if (list(idx2).count(True)) <= 1:
	# 				continue
	# 			Y2 = Y1[:, idx1==True]
	# 			Y2[Y2==0] = math.inf
	# 			min_val = np.min(np.array(Y2))
	# 			# ...
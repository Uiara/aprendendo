
class Node:
    
    def __init__(self, dado, anterior = None, proximo = None):
		self._dado = dado 
		self._proximo = proximo
		self._anterior = anterior

	def getDado(cls):
		return cls._dado

	def setDado(cls, novoDado):
		''' Este método recebe outro nó como parâmetro. '''
		cls._dado = novoDado

	def getAnterior(cls):
		return cls._anterior

	def setAnterior(cls, novoAnterior):
		''' Este método recebe outro nó como parâmetro. '''
		cls._anterior = novoAnterior

	def getProx(cls):
		return cls._proximo

	def setProx(cls, novoProx):
		''' Este método recebe outro nó como parâmetro. '''
		cls._proximo = novoProx


class Pilha:
	def __init__(self, conteudo, inicio = None, fim = None):
		self._inicio = inicio
		self._fim = fim 
		self.conteudo = conteudo 	# Recebe uma lista
	
	def getInicio(cls):
		return cls._inicio

	def setInicio(cls, novoInicio):
		''' Recebe um nó como parâmetro. '''
		cls._inicio = novoInicio

	def getFim(cls):
		''' Retorna o último nó. '''
		return cls._fim

	def setFim(cls, novoFim):
		''' Recebe um nó como parâmetro. '''
		cls._fim = novoFim

	def isVazia(cls):
		return (cls.getInicio() is None) and (cls.getFim() is None)

	def insereNoFim(cls):
		''' Insere um nó após o último elemento da lista. '''
		for item in cls.conteudo:
			no = Node(item)	# criando um nó com um dado específico
			if cls.isVazia():
				cls.setInicio(no)
				cls.setFim(no)
			else:
				no.setAnterior(cls.getFim())
				cls.getFim().setProx(no)
				cls.setFim(no)

	def removerUltimo(cls):
		''' Remove o último elemento da lista e retorna o dado deste elemento. '''
		if not cls.isVazia():
			dadoRemovido = cls.getFim().getDado()
			if cls.getInicio() == cls.getFim():
				# Caso a lista só tenha um único elemento.
				cls.setInicio(None) 
				cls.setFim(None)
			else:
				# Caso ela esteja multipopulada.
				cls.setFim(cls.getFim().getAnterior())
				cls.getFim().setProx(None)
			return dadoRemovido
		else:
			print('Lista vazia, cão!')

	def varrer(cls):
		''' Varredura da pilha para dizer se a entrada é válida. '''
		contParenteses = 0
		contColchetes = 0
		contChaves = 0

		while not cls.isVazia() and (contParenteses >= 0 and contColchetes >= 0 and contChaves >= 0):
			exp = cls.removerUltimo()
			if exp == ')': 
				contParenteses += 1
			elif exp == '(':
				contParenteses -= 1
			elif exp == ']':
				contColchetes += 1
			elif exp == '[':
				contColchetes -= 1
			elif exp == '}':
				contChaves += 1
			elif exp == '{':
				contChaves -= 1

		if contParenteses == 0 and contColchetes == 0 and contChaves == 0:
			return 'S'
		else:
			return 'N'


# RESULTADOS DAS EXPRESSÕES:

verificacoes = int(input())
listaExpressoes = []
for i in range(verificacoes):
	entrada = input()
	listaExpressoes += [entrada]
	
for expressao in listaExpressoes:
	pilha = Pilha(expressao)
	pilha.insereNoFim()
	print(pilha.varrer())
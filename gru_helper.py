#coding: utf-8
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.modules.rnn import GRU
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from .cuda_helper import zeros, Tensor, LongTensor
from .gumbel import gumbel_max
from .storage import Storage

F_GRUCell = torch._C._VariableFunctions.gru_cell

def sortSequence(data, length):
	shape = data.shape
	len, fsize = shape[0], shape[-1]
	data = data.reshape(len, -1, fsize)
	batch_size = data.shape[1]
	length = length.reshape(-1)

	zero_num = np.sum(length == 0)
	memo = list(reversed(np.argsort(length).tolist()))[:batch_size-zero_num]
	res = zeros(data.shape[0], batch_size - zero_num, data.shape[-1])
	for i, idx in enumerate(memo):
		res[:, i, :] = data[:, idx, :]
	return res, sorted(length, reverse=True)[: batch_size - zero_num], (shape, memo, zero_num)

def sortSequenceByMemo(data, memo):
	data = data.reshape(-1, data.shape[-1])
	batch_size = data.shape[0]
	shape, memo, zero_num = memo
	res = zeros(batch_size - zero_num, data.shape[-1])
	for i, idx in enumerate(memo):
		res[i, :] = data[idx, :]
	return res

def revertSequence(data, memo, isseq=False):
	shape, memo, zero_num = memo
	if isseq:
		res = zeros(data.shape[0], data.shape[1]+zero_num, data.shape[2])
		for i, idx in enumerate(memo):
			res[:, idx, :] = data[:, i, :]
		return res.reshape(*((res.shape[0], )+shape[1:-1]+(res.shape[-1], )))
	else:
		res = zeros(data.shape[0]+zero_num, data.shape[1])
		for i, idx in enumerate(memo):
			res[idx, :] = data[i, :]
		return res.reshape(*(shape[1:-1]+(res.shape[-1], )))

def flattenSequence(data, length):
	arr = []
	for i in range(length.size):
		arr.append(data[0:length[i], i])
	return torch.cat(arr, dim=0)

def copySequence(data, length): # for BOW loss
	arr = []
	for i in range(length.size):
		arr.append(data[i].repeat(length[i], 1))
	return torch.cat(arr, dim=0)

def generateMask(seqlen, length, type=int):
	return Tensor(
		(np.expand_dims(np.arange(seqlen), 1) < np.expand_dims(length, 0)).astype(type))

def maskedSoftmax(data, length):
	mask = generateMask(data.shape[0], length)
	return data.masked_fill(mask == 0, -1e9).softmax(dim=0)

def maskedLogSoftmax(data, length):
	mask = generateMask(data.shape[0], length)
	return torch.log_softmax(data.masked_fill(mask == 0, -1e9), dim=0)

class MyGRU(nn.Module):
	def __init__(self, input_size, hidden_size, layers=1, bidirectional=False, initpara=True):
		super(MyGRU, self).__init__()

		self.input_size, self.hidden_size, self.layers, self.bidirectional = \
				input_size, hidden_size, layers, bidirectional
		self.GRU = GRU(input_size, hidden_size, layers, bidirectional=bidirectional)
		self.initpara = initpara
		if initpara:
			if bidirectional:
				self.h_init = Parameter(torch.Tensor(2 * layers, 1, hidden_size))
			else:
				self.h_init = Parameter(torch.Tensor(layers, 1, hidden_size))
		self.reset_parameters()

	def reset_parameters(self):
		if self.initpara:
			stdv = 1.0 / math.sqrt(self.hidden_size)
			self.h_init.data.uniform_(-stdv, stdv)

	def getInitialParameter(self, batch_size):
		return self.h_init.repeat(1, batch_size, 1)

	def forward(self, incoming, length, h_init=None, need_h=False):
		sen_sorted, length_sorted, memo = sortSequence(incoming, length)
		left_batch_size = sen_sorted.shape[-2]
		sen_packed = pack_padded_sequence(sen_sorted, length_sorted)
		if h_init is None:
			h_init = self.getInitialParameter(left_batch_size)
		else:
			h_init = torch.unsqueeze(sortSequenceByMemo(h_init, memo), 0)
		h, h_n = self.GRU(sen_packed, h_init)
		h_n = h_n.transpose(0, 1).reshape(left_batch_size, -1)
		h_n = revertSequence(h_n, memo)
		if need_h:
			h = pad_packed_sequence(h)[0]
			h = revertSequence(h, memo, True)
			return h_n, h
		else:
			return h_n, None

class SingleGRU(nn.Module):
	def __init__(self, input_size, hidden_size, initpara=True):
		super().__init__()

		self.input_size, self.hidden_size = input_size, hidden_size
		self.GRU = GRU(input_size, hidden_size, 1)
		self.initpara = initpara
		if initpara:
			self.h_init = Parameter(torch.Tensor(1, 1, hidden_size))
			stdv = 1.0 / math.sqrt(self.hidden_size)
			self.h_init.data.uniform_(-stdv, stdv)

	def getInitialParameter(self, batch_size):
		return self.h_init.repeat(1, batch_size, 1)

	def forward(self, incoming, length, h_init=None, need_h=False):
		sen_sorted, length_sorted, memo = sortSequence(incoming, length)
		left_batch_size = sen_sorted.shape[-2]
		sen_packed = pack_padded_sequence(sen_sorted, length_sorted)
		if h_init is None:
			h_init = self.getInitialParameter(left_batch_size)
		else:
			h_init = torch.unsqueeze(sortSequenceByMemo(h_init, memo), 0)
		h, h_n = self.GRU(sen_packed, h_init)
		h_n = h_n.transpose(0, 1).reshape(left_batch_size, -1)
		h_n = revertSequence(h_n, memo)
		if need_h:
			h = pad_packed_sequence(h)[0]
			h = revertSequence(h, memo, True)
			return h_n, h
		else:
			return h_n, None

	def init_forward(self, batch_size, h_init=None):
		if h_init is None:
			h_init = self.getInitialParameter(batch_size)
		else:
			h_init = torch.unsqueeze(h_init, 0)
		h_history = h_init
		h = h_init[0]

		def nextStep(incoming, stopmask):
			nonlocal h_history, h
			h = self.cell_forward(incoming, h) * (1 - stopmask).float().unsqueeze(-1)
			return h

		return nextStep

	def cell_forward(self, incoming, h):
		return F_GRUCell( \
				incoming, h, \
				self.GRU.weight_ih_l0, self.GRU.weight_hh_l0, \
				self.GRU.bias_ih_l0, self.GRU.bias_hh_l0, \
		)

	def freerun(self, inp, wLinearLayerCallback, mode='max', input_callback=None):
		# batch_size, dm, embLayer, max_sent_length, [init_h]
		# w emb length
		batch_size = inp.batch_size
		dm = inp.dm

		first_emb = inp.embLayer(LongTensor([dm.go_id])).repeat(batch_size, 1)

		gen = Storage()
		gen.w_pro = []
		gen.w_o = []
		gen.emb = []
		flag = zeros(batch_size).byte()
		EOSmet = []

		next_emb = first_emb
		nextStep = self.init_forward(batch_size, inp.get("init_h", None))

		for i in range(inp.max_sent_length):
			now = next_emb
			if input_callback:
				now = input_callback(i, now)

			gru_h = nextStep(now, flag)
			w = wLinearLayerCallback(gru_h)
			gen.w_pro.append(w)
			if mode == "max":
				w = torch.argmax(w[:,2:], dim=1) + 2
				next_emb = inp.embLayer(w)
			elif mode == "gumbel":
				w_onehot, w = gumbel_max(w[:,2:], 1, 1)
				w = w + 2
				next_emb = torch.sum(torch.unsqueeze(w_onehot, -1) * inp.embLayer.weight[2:], 1)
			gen.w_o.append(w)
			gen.emb.append(next_emb)

			EOSmet.append(flag)
			flag = flag | (w == dm.eos_id)
			if torch.sum(flag).detach().cpu().numpy() == batch_size:
				break

		EOSmet = 1-torch.stack(EOSmet)
		gen.w_o = torch.stack(gen.w_o) * EOSmet.long()
		gen.emb = torch.stack(gen.emb) * EOSmet.float().unsqueeze(-1)
		gen.length = torch.sum(EOSmet, 0).detach().cpu().numpy()

		return gen


class SingleAttnGRU(nn.Module):
	def __init__(self, input_size, hidden_size, post_size, initpara=True, gru_input_attn=False):
		super().__init__()

		self.input_size, self.hidden_size, self.post_size = \
			input_size, hidden_size, post_size
		self.gru_input_attn = gru_input_attn

		if self.gru_input_attn:
			self.GRU = GRU(input_size + post_size, hidden_size, 1)
		else:
			self.GRU = GRU(input_size, hidden_size, 1)

		self.attn_query = nn.Linear(hidden_size, post_size)

		if initpara:
			self.h_init = Parameter(torch.Tensor(1, 1, hidden_size))
			stdv = 1.0 / math.sqrt(self.hidden_size)
			self.h_init.data.uniform_(-stdv, stdv)

	def forward(self, incoming, length, post, post_length, h_init=None):
		batch_size = incoming.shape[1]
		seqlen = incoming.shape[0]
		if h_init is None:
			h_init = self.getInitialParameter(batch_size)
		else:
			h_init = torch.unsqueeze(h_init, 0)
		h_now = h_init[0]
		hs = []
		attn_weights = []
		context = zeros(batch_size, self.post_size)

		for i in range(seqlen):
			if self.gru_input_attn:
				h_now = self.cell_forward(torch.cat([incoming[i], context], last_dim=-1), h_now) \
					* Tensor((length > np.ones(batch_size) * i).astype(float)).unsqueeze(-1)
			else:
				h_now = self.cell_forward(incoming[i], h_now) \
					* Tensor((length > np.ones(batch_size) * i).astype(float)).unsqueeze(-1)

			query = self.attn_query(h_now)
			attn_weight = maskedSoftmax((query.unsqueeze(0) * post).sum(-1), post_length)
			context = (attn_weight.unsqueeze(-1) * post).sum(0)

			hs.append(torch.cat([h_now, context], dim=-1))
			attn_weights.append(attn_weight)

		return h_now, hs, attn_weights

	def init_forward(self, batch_size, post, post_length, h_init=None):
		if h_init is None:
			h_init = self.getInitialParameter(batch_size)
		else:
			h_init = torch.unsqueeze(h_init, 0)
		h_now = h_init[0]
		context = zeros(batch_size, self.post_size)

		def nextStep(incoming, stopmask):
			nonlocal h_now, post, post_length, context
			
			if self.gru_input_attn:
				h_now = self.cell_forward(torch.cat([incoming, context], dim=-1), h_now) \
					* (1 - stopmask).float().unsqueeze(-1)
			else:
				h_now = self.cell_forward(incoming, h_now) * (1 - stopmask).float().unsqueeze(-1)

			query = self.attn_query(h_now)
			attn_weight = maskedSoftmax((query.unsqueeze(0) * post).sum(-1), post_length)
			context = (attn_weight.unsqueeze(-1) * post).sum(0)

			return torch.cat([h_now, context], dim=-1), attn_weight

		return nextStep

	def cell_forward(self, incoming, h):
		return F_GRUCell( \
				incoming, h, \
				self.GRU.weight_ih_l0, self.GRU.weight_hh_l0, \
				self.GRU.bias_ih_l0, self.GRU.bias_hh_l0, \
		)

class SingleSelfAttnGRU(nn.Module):
	def __init__(self, input_size, hidden_size, attn_wait=3, initpara=True):
		super().__init__()

		self.input_size, self.hidden_size = \
				input_size, hidden_size
		self.attn_wait = attn_wait
		self.decoderGRU = GRU(input_size + hidden_size, hidden_size, 1)
		self.encoderGRU = GRU(input_size, hidden_size, 1)

		self.attn_query = nn.Linear(hidden_size, hidden_size)

		#self.attn_null = Parameter(torch.Tensor(1, 1, hidden_size))
		#stdv = 1.0 / math.sqrt(self.hidden_size)
		#self.attn_null.data.uniform_(-stdv, stdv)

		if initpara:
			self.eh_init = Parameter(torch.Tensor(1, 1, hidden_size))
			stdv = 1.0 / math.sqrt(self.hidden_size)
			self.eh_init.data.uniform_(-stdv, stdv)
			self.dh_init = Parameter(torch.Tensor(1, 1, hidden_size))
			self.dh_init.data.uniform_(-stdv, stdv)

	def forward(self, incoming, length, eh_init=None, dh_init=None, need_h=False, need_attn_weight=False):
		batch_size = incoming.shape[1]
		seqlen = incoming.shape[0]

		if eh_init is None:
			eh_init = self.eh_init.repeat(1, batch_size, 1)
		else:
			eh_init = torch.unsqueeze(eh_init, 0)
		if dh_init is None:
			dh_init = self.dh_init.repeat(1, batch_size, 1)
		else:
			dh_init = torch.unsqueeze(dh_init, 0)

		h_history = []
		eh = eh_init[0]
		dh = dh_init[0]
		dhs = []
		attn_weights = []
		#attn_null = self.attn_null.repeat(1, batch_size, 1)
		for i in range(seqlen):
			if i <= self.attn_wait:
				context = zeros(batch_size, self.hidden_size)
			else:
				query = self.attn_query(dh)
				h_wait = h_history[:self.attn_wait]
				attn_weight = (query.unsqueeze(0) * h_wait).sum(-1).softmax(0)
				attn_weights.append(attn_weight)
				context = (attn_weight.unsqueeze(-1) * h_wait).sum(0)
			dh = F_GRUCell(
				torch.cat([incoming[i], context], dim=-1), dh,
				self.decoderGRU.weight_ih_l0, self.decoderGRU.weight_hh_l0,
				self.decoderGRU.bias_ih_l0, self.decoderGRU.bias_hh_l0
			) * Tensor((length > np.ones(batch_size) * i).astype(float)).unsqueeze(-1)

			eh = F_GRUCell(
				incoming[i], eh,
				self.encoderGRU.weight_ih_l0, self.encoderGRU.weight_hh_l0,
				self.encoderGRU.bias_ih_l0, self.encoderGRU.bias_hh_l0
			) * Tensor((length > np.ones(batch_size) * i).astype(float)).unsqueeze(-1)

			h_history = eh.unsqueeze(0) if not h_history else torch.cat([h_history, eh.unsqueeze(0)], dim=0)
			dhs.append(dh)

		h_n = dh
		if need_h:
			h = torch.stack(dhs, 0)
			if need_attn_weight:
				return h, h_n, attn_weights
			else:
				return h, h_n
		else:
			return h_n

	def init_forward(self, batch_size, eh_init=None, dh_init=None):
		if eh_init is None:
			eh_init = self.eh_init.repeat(1, batch_size, 1)
		else:
			eh_init = torch.unsqueeze(eh_init, 0)
		if dh_init is None:
			dh_init = self.dh_init.repeat(1, batch_size, 1)
		else:
			dh_init = torch.unsqueeze(dh_init, 0)

		h_history = []
		dh = dh_init[0]
		eh = eh_init[0]
		#attn_null = self.attn_null.repeat(1, batch_size, 1)

		def nextStep(incoming, stopmask):
			nonlocal h_history, eh, dh

			if h_history is None or h_history.shape[0] <= self.attn_wait:
				context = zeros(batch_size, self.hidden_size)
			else:
				query = self.attn_query(dh)
				h_wait = h_history[:self.attn_wait]
				attn_weight = (query.unsqueeze(0) * h_wait).sum(-1).softmax(0)
				context = (attn_weight.unsqueeze(-1) * h_wait).sum(0)

			dh = F_GRUCell(
				torch.cat([incoming, context], dim=-1), dh,
				self.decoderGRU.weight_ih_l0, self.decoderGRU.weight_hh_l0,
				self.decoderGRU.bias_ih_l0, self.decoderGRU.bias_hh_l0
			) * (1 - stopmask).float().unsqueeze(-1)

			eh = F_GRUCell(
				incoming, eh,
				self.encoderGRU.weight_ih_l0, self.encoderGRU.weight_hh_l0,
				self.encoderGRU.bias_ih_l0, self.encoderGRU.bias_hh_l0
			) * (1 - stopmask).float().unsqueeze(-1)
			h_history = eh.unsqueeze(0) if not h_history else torch.cat([h_history, eh.unsqueeze(0)], dim=0)
			return dh

		return nextStep

import torch
torch.manual_seed(1234)
a = torch.randint(1, 10, (2, 2, 4))
b = torch.randint(1, 5, (2, 2, 4))
# print(object(a), separator=separator, end=end, file=file, flush=flush)

# b = torch.randn(2,4)
# print(b)
# i = torch.max(b,1)[1][1]
# print(int(i))
value, index = torch.topk(a, 3, dim=-1)
list_index = [1, 2, 4]
# mask = a[][][]
# print(a, b)
# print(value)
# print(index)
# print(index.size())
# # print(torch.gather(a, dim=-1, index=index))
# # print(a[:,:,index.squeeze(0).squeeze(1)])
# print(a[:, :, list_index])


# a = torch.tensor([
#     [[6, 1, 3, 4], [2, 3, 7, 2]],
#     [[5, 2, 1, 3], [6, 8, 2, 8]]
# ])
# b = torch.tensor([
#     [[5, 1, 2, 0, 6, 3, 4, 2], [6, 3, 8, 6, 1, 3, 8, 2]],
#     [[2, 4, 8, 7, 1, 2, 0, 8], [2, 5, 7, 8, 4, 1, 6, 3]]
# ])
# index = torch.tensor([0, 1, 1, 2, 2, 2, 3, 3])
# index2 = torch.tensor([
#     [[0, 1, 1, 2, 2, 2], [0, 1, 1, 2, 2, 2]],
#     [[0, 1, 1, 2, 2, 2], [0, 1, 1, 2, 2, 2]]
# ])
# dim1, dim2, dim3 = a.size()
# print(a.size())
# print(b.size())

# print(index.repeat(b.size()[0], b.size()[1]).reshape(dim1, dim2, -1))
# print(torch.gather(a, 2, index2) + b)
# print(a[:, :, index])
# print(a[:, :, index] + b)
c = torch.Tensor([
    [6, 1, 0, 0], [2, 3, 7, 0],
    [5, 0, 0, 0], [6, 8, 4, 0]
])

g = torch.Tensor([
    [[6, 1, 0], [2, 3, 7]],
    [[1, 9, 2], [7, 8, 3]],
    [[5, 0, 0], [6, 8, 4]],
    [[5, 4, 3], [6, 9, 2]]
])
g1 = torch.Tensor([
    [[6, 1, 0], [2, 3, 7]],
    [[1, 9, 2], [7, 8, 3]],
    [[5, 0, 0], [6, 8, 4]],
    [[5, 4, 3], [6, 9, 2]]
])
d = torch.Tensor([
    [1, 9, 3, 2], [7, 3, 8, 3],
    [5, 0, 4, 3], [6, 9, 2, 2]
])

e = torch.Tensor([
    [2, 6, 1, 6, 4, 6, 4, 6, 9, 1, 4, 3, 4, 6, 3, 5], [7, 3, 2, 0, 4, 3, 8, 1, 6, 4, 6, 3, 2, 0, 4, 3],
    [5, 0, 4, 3, 4, 6, 9, 1, 0, 4, 3, 8, 6, 4, 6, 9], [6, 9, 2, 8, 3, 2, 0, 2, 0, 4, 2, 0, 4, 3, 3, 4]
])
# print(g.shape)
# mask = c.gt(0)
# mask = mask.repeat(2, 4, 4)
# print(mask)
# mask = mask.transpose(0, 1)
# print(mask)

# print(g.permute(1, 0, 2), g.permute(1, 0, 2).shape)
# print(g.transpose(0, 1), g.transpose(0, 1).shape)
t , sorted_index = torch.sort(c, dim=-1,descending=True)
_, max_index = torch.max(c, dim=-1)
pre= torch.argmax(c, dim=-1)
print(sorted_index[:, 0])
print(pre)
print(max_index)
# print(c.tolist())
# print(sorted_index)
# sorted_index = sorted_index.lt(2)
# print(sorted_index)
# score, index = torch.topk(c, k=2, dim=-1, sorted=True)
# print(score)
# maxk_score = torch.min(score,dim=-1)[0]
# print(maxk_score.size())
# print(c)
# print(c.ge(maxk_score))
# print(maxk_score.view(4,1))
# print(torch.where(c.ge(maxk_score.view(4,1)), c, torch.full_like(c, 0)))
# print(index)
# score2, index2 = torch.topk(d, k=2, dim=-1, sorted=True)
# print(index2)
# z = torch.zeros(4, 4).scatter_(1, torch.LongTensor(([[0, 3], [2, 1], [0, 3], [1, 3]])), c)
# print(z)
# print(score, '\n', index)
# print(score2, '\n', index2)
# print(score.size())
# print(index.size())
# mask = score.gt(4)
# print(mask)
# print(torch.masked_select(index, mask))

# e = torch.nonzero(torch.Tensor([[0.6, 0.5, 0.0, 0.0],
#                              [0.0, 0.4, 0.0, 0.0],
#                              [0.0, 0.0, 1.2, 0.0],
#                              [0.0, 0.0, 0.0,-0.4]]))
# print(e)

# a = torch.Tensor([6,1,3,4])
# b = torch.Tensor([1,9,3,2])
# score1, index1 = torch.topk(a, k=2, dim=-1, sorted=False)
# score2, index2 = torch.topk(b, k=2, dim=-1, sorted=False)
# min_score1 = torch.min(score1)
# min_score2 = torch.min(score2)
# mask1 = a.ge(min_score1)
# print(mask1)
# mask2 = b.ge(min_score2)
# print(mask2)
# print(torch.masked_select(a, mask1))


# mask = c.gt(0)
# sen_lens = mask.sum(1)
# print(sen_lens)
# print(torch.split(c[mask], sen_lens.tolist()))


# y = [1,-1,2,4,-1]
# print(list(filter(lambda x : x!=-1, y)))

# def is_legal_label(self, label, pre_label):
#         ws_label = pre_label.split('-')[0] + '_' + label.split('-')[0]
#         if ws_label in self.illegal_bies:
#             return False
#         elif ws_label in self.continuous_bies:
#             if pre_label.split('-')[1] != label.split('-')[1]:
#                 return False
#             else:
#                 return True
#         else:
#             return True

# def is_legal_coupled_label(self, label, pre_label):
#     label1, label2 = label.split('@')
#     pre_label1, pre_label2 = pre_label.split('@')
#     return (self.is_legal_label(label1, pre_label1) and self.is_legal_label(label2, pre_label2))

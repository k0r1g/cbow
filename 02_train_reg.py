#
#
#
import torch
import model


#
#
#
cbow = model.CBOW(63642, 128)
cbow.load_state_dict(torch.load('./checkpoints/2025_04_17__11_04_09.5.cbow.pth'))
cbow.eval()


#
#
#
mReg = model.Regressor()
opFoo = torch.optim.Adam(mReg.parameters(), lr=0.005)


#
#
#
for i in range(100):
  trg = torch.tensor([[125.]])               # score
  ipt = torch.tensor([[45, 27, 45367, 456]]) # title
  emb = cbow.emb(ipt).mean(dim=1)            # avg pool
  out = mReg(emb)
  loss = torch.nn.functional.l1_loss(out, trg)
  loss.backward()
  opFoo.step()
  opFoo.zero_grad()
  print(loss.item())

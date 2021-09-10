import torch
from Modules import Gnet, Dnet
from torch import optim, nn
from torch.utils.data import DataLoader
from Datasets import MNIST_dataset
from torchvision.utils import save_image

class Trainer():
    def __init__(self, batch_size, root):
        self.batch_size = batch_size
        root = root
        dataset = MNIST_dataset(root)
        self.datalodar = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)


        self.dnet = Dnet().cuda()
        self.gnet = Gnet().cuda()
        self.opt_d = optim.Adam(self.dnet.parameters(), lr=0.0009, betas=(0.5, 0.999))
        #self.scheduler_d = optim.lr_scheduler.ExponentialLR(self.opt_d, gamma=0.099)
        self.opt_g = optim.Adam(self.gnet.parameters(), lr=0.001, betas=(0.5, 0.999))

        self.loss = nn.BCELoss()
        # self.recloss = nn.MSELoss()


    def __call__(self):
        k = 0
        for epoch in range(1000):
            sum_dloss = 0
            sum_gloss = 0
            for i, (img, lable) in enumerate(self.datalodar):
                # print("img", img.shape)
                # print("lable", lable.shape)

                img = img.reshape(-1, 1, 28, 28)
                img = img.cuda()
                lable = lable.cuda()

                real_lable = torch.ones(img.shape[0], dtype=torch.float32).cuda()
                fake_lable = torch.zeros(img.shape[0], dtype=torch.float32).cuda()

                noise_d = torch.normal(0, 0.02, (img.shape[0], 64, 1, 1), dtype=torch.float32).cuda()

                real_out = self.dnet(img, lable)
                real_lable_out = self.loss(real_out, real_lable)
                fake_in = self.gnet(noise_d, lable)
                fake_out = self.dnet(fake_in, lable)
                fake_lable_out = self.loss(fake_out, fake_lable)
                d_loss = real_lable_out + fake_lable_out
                sum_dloss += d_loss.item()

                self.opt_d.zero_grad()
                d_loss.backward()
                self.opt_d.step()
                #self.scheduler_d.step()

                noise_g = torch.normal(0, 0.02, (img.shape[0], 64, 1, 1), dtype=torch.float32).cuda()
                g_out = self.gnet(noise_g, lable)
                g_2_d = self.dnet(g_out, lable)
                g_2_real = self.loss(g_2_d, real_lable)
                self.opt_g.zero_grad()
                g_2_real.backward()
                self.opt_g.step()
                sum_gloss += g_2_real.item()

                fake_img = self.gnet(noise_g, lable)

                if i % 100 == 0:
                    img = img.reshape(-1, 1, 28, 28)
                    fake_img = fake_img.reshape(-1, 1, 28, 28)
                    save_image(img, "img/{}-real_img.jpg".format(k + 1), nrow=10)
                    save_image(fake_img, "img/{}-fake_img.jpg".format(k + 1), nrow=10)
                k += 1

            avg_dloss = sum_dloss / len(self.datalodar)
            avg_gloss = sum_gloss / len(self.datalodar)
            print(epoch, "avg_dloss", avg_dloss)
            print(epoch, "avg_gloss", avg_gloss)
            if epoch % 10 == 9:
                torch.save(self.dnet.state_dict(), "weight/{}-dwight.pth".format(epoch))
                torch.save(self.gnet.state_dict(), "weight/{}-gwight.pth".format(epoch))



if __name__ == '__main__':
    train = Trainer(root=r"D:\Data_set\MNIST_IMG", batch_size=200)
    train()


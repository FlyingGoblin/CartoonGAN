cd checkpoints
#wget http://cg.cs.tsinghua.edu.cn/people/~Yongjin/CartoonGAN-Models.rar
#rar x CartoonGAN-Models.rar

mkdir Hayao
mv CartoonGAN-Models/Hayao_net_G.t7 Hayao/latest_net_G.t7

mkdir Hosoda
mv CartoonGAN-Models/Hosoda_net_G.t7 Hosoda/latest_net_G.t7

mkdir Paprika
mv CartoonGAN-Models/Paprika_net_G.t7 Paprika/latest_net_G.t7

mkdir Shinkai
mv CartoonGAN-Models/Shinkai_net_G_A.t7 Shinkai/latest_net_G.t7

#rm -r CartoonGAN-Models
#rm CartoonGAN-Models.rar

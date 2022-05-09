# skipnet-adversarial-training
Adversarial Training for Robustness of Dynamic Neural Networks

First place adversarial images in a folder <adv_images>.

To recreate results in project report, run train_sp.py with appropriate arguments. Eg:

<code> python3 train_sp.py train cifar10_rnn_gate_38 --resume resnet-38-rnn-sp-cifar10.pth.tar --pretrained -d adversarial --gate-type rnn --iters 40000 --eval-every 250 --lr 0.0001 --weight-decay 0 --size=1250 --train-alphas="0.0,0.5,0.9,0.99,0.999,1.0" --test-alphas="0.0,0.5,0.9,0.99,0.999,1.0" --image-dir "output3" --full-train </code>

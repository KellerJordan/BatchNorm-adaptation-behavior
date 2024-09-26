mask = (torch.randperm(50000) < 10000)
for k in tqdm(corruptions):
    obj = torch.load('/home/ubuntu/cifar10-zoo/adapation/shifts/cifar-10-c/%s/test.pt' % k)
    x = obj['images'][mask].clone()
    y = obj['labels'][mask].clone()
    obj = dict(images=x, labels=y, classes=obj['classes'])
    p = '/home/ubuntu/BatchNorm-adaptation/corruption_data/%s' % k
    os.makedirs(p, exist_ok=True)
    torch.save(obj, '%s/test.pt' % p)

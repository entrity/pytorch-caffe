# Usage
# 
# Use Python 2
# caffenet will not work with Python 3

import sys, os, torch
srcdir = sys.argv[1] # conv.prototxt

import caffenet

def do(fname):
	net = caffenet.CaffeNet(os.path.join(srcdir, fname))
	net.load_weights(os.path.join(srcdir, 'vgg16_fast_rcnn_iter_80000.caffemodel'))
	torch.save(net.state_dict(), '%s.statedict.pth' % fname)
	torch.save(net, '%s.full.pth' % fname)

do('conv.prototxt')
do('linear.prototxt')

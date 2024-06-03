import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from subprocess import Popen, PIPE

#################### visdom function #############################################

class Visualizer():
    def __init__(self, display_id=1, address="172.16.33.116", port=8097):    
        """Initialize the Visualizer class
        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test rver
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.display_id = display_id
        self.display_server = address
        # self.win_size = 
        self.name = "debug"
        self.port = port
        # self.saved = False
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = 2
            self.vis = visdom.Visdom(server=self.display_server, port=self.port)
            if not self.vis.check_connection():
                print('what?')
                self.create_visdom_connections()


    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server' # + ' -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, win=None):
        if self.display_id <= 0:
            return
        title = self.name
        images = []
        labels = []

        ncols = min(self.ncols, len(visuals))
        h, w = next(iter(visuals.values())).shape[:2]

        table_css = """<style>
                table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                </style>""" % (w, h)  # create a table css
        label_html = ''
        label_html_row = ''
        
        if win is None:
            idx = 0
        else:
            idx = win

        for label, image in visuals.items():
            # images.append(image)
            # labels.append(label)
            idx += 1

            if image.min() < 0:
                image = (image + 1) / 2

            self.vis.images(image, nrow=1, win=self.display_id + idx,
                            padding=2, opts=dict(title=label + ' images'))    
            # label_html_row += '<td>%s</td>' % label
            # if idx % ncols == 0:
            #     label_html += '<tr>%s</tr>' % label_html_row
            #     label_html_row = ''
        # while idx % ncols != 0:
        #     images.append(white_image)
        #     label_html_row += '<td></td>'
        #     idx += 1
        #     if label_html_row != '':
        #         label_html += '<tr>%s</tr>' % label_html_row

        # try:
        #     self.vis.images(images, nrow=ncols, win=self.display_id + 1,
        #                     padding=2, opts=dict(title=title + ' images'))        
        #     self.vis.text(table_css + label_html, win=self.display_id + 2,
        #                   opts=dict(title=title + ' labels'))
        # except Exception as e:
        #     print(e)
            # self.create_visdom_connections()

    def plot_current_losses(self, epoch, counter_ratio, losses, title_name='loss',display_id=None):
        """display the current losses on visdom display: dictionary of error labels and values
        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if self.display_id <= 0:
            return
        
        display_id = self.display_id if display_id is None else display_id
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])

        # X = np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1)
        # Y = np.array(self.plot_data['Y'])

        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + f' {title_name} over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=display_id)

    def yell(self, text):
        if not hasattr(self, 'text'):
            self.text = self.vis.text(text)
        else:
            self.text = self.vis.text(text, win=self.text)


########################################################################################

def tensor_to_visual(img, normalize=False):
    signed_to_unsigned = lambda x : (x + 1) / 2
    img = img[0].detach().cpu().numpy()
    if img.min() < 0:
        img = signed_to_unsigned(img)
    # img = (img - img.min()) / (img.max() - img.min())    
    if normalize:
        img = (img - img.min()) / (img.max() - img.min() + 1e-4)
    img = np.clip(img,0.0,1.0)
    return img


def get_checkerboard(size):
    a = np.arange(size)
    x, y = np.meshgrid(a, a)
    grid = np.stack([x, y])

    x = (grid[0] / 16).astype(np.uint8) % 2
    y = (grid[1] / 16).astype(np.uint8) % 2
    checker = np.logical_xor(x, y)[..., np.newaxis]

    color1 = checker * np.array([0.6, 0.6, 0.6]).reshape(1, 1, 3)
    color2 = np.invert(checker) * np.array([0.7, 0.7, 0.7]).reshape(1, 1, 3)

    color = color1 + color2
    return color

def slice_volume(voxels, slice_idx, is_batch=True, flip=True):
    slice = voxels[:, :, slice_idx, :, :]
    show_image(slice, is_batch=is_batch, flip=flip)

def pytorch_to_numpy(array, is_batch=True, flip=True):
    array = array.detach().cpu().numpy()

    if flip:
        source = 1 if is_batch else 0
        dest = array.ndim - 1
        array = np.moveaxis(array, source, dest)

    return array

def show_image(image, is_batch=True, flip=True, resolution=-1, path=None):
    if resolution != -1:
        image = torch.nn.functional.interpolate(image, size=(resolution, resolution), mode='nearest')

    if type(image).__module__ == 'torch':
        image = pytorch_to_numpy(image, is_batch, flip)
    bs = image.shape[0]
    size = image.shape[1]
    channel = image.shape[-1]

    if channel == 2:
        image = np.concatenate((image, np.zeros_like(image[...,0:1])), axis=-1)

    # if channel == 4:
    #     # add checkerboard
    #     checker = get_checkerboard(size)
    #     alpha = image[..., [3]]
    #     rgb = image[..., 0:3]
    #     image = checker * (1 - alpha) + rgb * alpha

    cmap = None

    if image.ndim == 3:
        fig, ax = plt.subplots()
        im = ax.imshow(np.squeeze(image), cmap=cmap)
        plt.colorbar(im, ax=ax)

    elif bs == 1:
        fig, ax = plt.subplots()
        im = ax.imshow(np.squeeze(image[0]), cmap=cmap)
        # plt.colorbar(im, ax=ax)
    else:
        fig, axes = plt.subplots(1, image.shape[0], figsize=(150, 150))
        for idx, axis in enumerate(axes):
            im = axis.imshow(np.squeeze(image[idx]), cmap=cmap)
            # plt.colorbar(im, ax=axis)

    if path is not None:
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight',transparent=True, pad_inches=0)
    else:
        plt.show()

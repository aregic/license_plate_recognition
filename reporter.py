import jinja2
import pdfkit
import os

"""
env = jinja2.Environment(loader=jinja2.FileSystemLoader('.'))
template = env.get_template("report_template.html")

template_vars = {'title' : 'THE TITLE', 'some_input' : 'THE SOME INPUT'}

html_out = template.render(template_vars)

pdfkit.from_string(html_out, "out.pdf")
"""
import ruamel.yaml
from inspect_images import draw_float_bounding_box
from data_feeder import *
from simple_nn import SimpleNet

config = ruamel.yaml.load(open("./simple_net_config.yaml"))
network_config = NetworkConfig(**config["SimpleNetConfig"])
env_config = EnvConfig(**config["EnvConfig"])

simple_net = SimpleNet(network_config, env_config)

image_iter = imageLabelIterator(env_config.sample_dir)
s = SlidingWindowSampleCreator(network_config.slide_x, network_config.slide_y, network_config.window_width,
                               network_config.window_height, normalize_label=True,
                               yes_label_weight=network_config.yes_label_weight,
                               no_label_weight=network_config.no_label_weight)

sliding_window_iter = s.create_sliding_window_from_iter(image_iter)

loss, image, orig_label, output, gradients = simple_net.train(sliding_window_iter, is_training=False)

def plot_result(i : int):
    draw_float_bounding_box(np.squeeze(image[i]), [orig_label[i][:4].reshape([2, 2])], [output[i][:4].reshape([2,2])])

datas = []
os.chdir('./report')
for i in range(len(output)):
    plot_result(i)
    image_loc = f'./images/figure{i}.png'
    plt.savefig(image_loc)
    datas.append({
        'orig_label' : orig_label[i][:4],
        'output' : output[i][:4],
        'image_loc' : image_loc[2:],
        'is_there_license_plate' : orig_label[i][4][0],
        'network_confidence' : output[i][4][0]
    })

env = jinja2.Environment(loader=jinja2.FileSystemLoader('.'))
template = env.get_template("report_template.html")
html_out = template.render({'title' : 'Network output', 'loss' : loss, 'datas' : datas})

pdfkit.from_string(html_out, 'report.pdf')

with open('report.html', 'w') as f:
    f.write(html_out)

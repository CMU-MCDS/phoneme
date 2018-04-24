import numpy as np
import math
import os
#import scipy.io.wavfile as wav
import soundfile as sf
import matplotlib.pyplot as plt


def main():
	folder_name = "../../phoneme_data/Griko/raw"

	duration_list = []
	for file_name in os.listdir(folder_name):
		print(file_name)
		#rate, snd = wav.read(folder_name + "/" + file_name)
		snd, rate = sf.read(folder_name + "/" + file_name)
		print(snd)
		print(rate)
		if len(snd.shape) == 1:
			snd_num = snd.shape[0]
			channel_num = 1
		else:
			snd_num, channel_num = snd.shape
		duration = float(snd_num) / rate
		duration_list.append(duration)
	duration_list = np.array(duration_list)

	total_dur = np.sum(duration_list)
	print("total duration", total_dur, " => in hours", total_dur / 3600)
	mean_dur = np.mean(duration_list)
	print("mean duration", mean_dur)
	stddev_dur = np.std(duration_list)
	print("stddev of duration", stddev_dur)


	"""
	t = np.array([float(n) / rate for n in range(snd_num)])

	# Subsample for plot
	SUBSAMPLE_INTERVAL = 100
	subsample_num = math.floor(snd_num / SUBSAMPLE_INTERVAL)
	print(subsample_num)
	t_plot = [t[n * SUBSAMPLE_INTERVAL] for n in range(subsample_num)]
	snd_1_plot = [snd_1[n * SUBSAMPLE_INTERVAL] for n in range(subsample_num)]
	snd_2_plot = [snd_2[n * SUBSAMPLE_INTERVAL] for n in range(subsample_num)]

	plt.figure(1)
	plt.plot(t_plot, snd_1_plot, 'r-', t_plot, snd_2_plot, 'b-')
	#plt.xlim([0, 11])
	#plt.ylim([0, 0.5])
	plt.xlabel('Time (sec)')
	plt.ylabel('Wave form')
	plt.savefig('Griko6.png')
	"""


if __name__ == "__main__":
	main()

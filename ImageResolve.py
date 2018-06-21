from PIL import Image
import struct


class ImageResolve():
    def __init__(self, sour_path=r'', dir_path=r''):
        self.sour_path = sour_path
        self.dir_path = dir_path

    def resolve(self):
        with open(self.sour_path, 'rb') as f:
            magic_num = struct.unpack('!i', f.read(4))[0]
            image_num = struct.unpack('!i', f.read(4))[0]
            rows = struct.unpack('!i', f.read(4))[0]
            cols = struct.unpack('!i', f.read(4))[0]

            for i in range(image_num):
                print(i)
                tmp_image = Image.new('L', (cols, rows))
                for r in range(rows):
                    for c in range(cols):
                        tmp_image.putpixel((c, r), struct.unpack('!B', f.read(1)))

                name = 'pic_%s.png' % str(i)
                tmp_image.save(self.dir_path + name)


class LableResolve():
    def __init__(self, sour_path=r'', dir_path=r''):
        self.sour_path = sour_path
        self.dir_path = dir_path

    def resolve(self):
        with open(self.sour_path, 'rb') as f:
            with open(self.dir_path, 'w') as f_w:
                magic_num = struct.unpack('!i', f.read(4))[0]
                lable_num = struct.unpack('!i', f.read(4))[0]

                for i in range(lable_num):
                    lable = struct.unpack('!B', f.read(1))[0]
                    f_w.writelines(str(lable) + '\n')


if __name__ == '__main__':
    ima_re = ImageResolve(sour_path=r'F:\研究生\zpc\神经网络课程\大作业\train-images-idx3-ubyte\train-images.idx3-ubyte',
                          dir_path=r'F:\研究生\zpc\神经网络课程\大作业\train_set\pic\\')
    ima_re.resolve()

    lable_re = LableResolve(sour_path=r'F:\研究生\zpc\神经网络课程\大作业\train-labels-idx1-ubyte\train-labels.idx1-ubyte',
                            dir_path=r'F:\研究生\zpc\神经网络课程\大作业\train_set\lable\train_lables.txt')
    lable_re.resolve()

    test_ima_re = ImageResolve(sour_path=r'F:\研究生\zpc\神经网络课程\大作业\t10k-images-idx3-ubyte\t10k-images.idx3-ubyte',
                               dir_path=r'F:\研究生\zpc\神经网络课程\大作业\test_set\pic\\')
    test_ima_re.resolve()

    test_lable_re = LableResolve(sour_path=r'F:\研究生\zpc\神经网络课程\大作业\t10k-labels-idx1-ubyte\t10k-labels.idx1-ubyte',
                                 dir_path=r'F:\研究生\zpc\神经网络课程\大作业\test_set\lable\test_lables.txt')
    test_lable_re.resolve()

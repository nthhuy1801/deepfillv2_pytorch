import cv2
import os
from matplotlib import image
from pyparsing import Opt
import torch
import numpy as np
from glob import glob
from models.model import Generator 
from torchvision.transforms import ToTensor


def postProcess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image

def demo(opt):
    # Load images
    img_list = []
    for ext in ['*.jpg', '*.png']:
        img_list.extend(glob(os.path.join(opt.test_dir, ext)))
    img_list.sort()

    # Load model
    model = Generator(opt)
    model.load_state_dict(torch.load(opt.pretrained, map_location='cpu'))
    model.eval()
    print("Loading pretrained model successful !!!")

    for fn in img_list:
        filename = os.path.basename(fn).split('.')[0]
        orig_img = cv2.resize(cv2.imread(fn, cv2.IMREAD_COLOR), (128,128))
        img_tensor = (ToTensor()(orig_img) * 2.0 - 1.0).unsqueeze(0)
<<<<<<< HEAD
        h, w, c  = orig_img.shape
        mask = np.zeros([h, w, 1], np.uint8)
        image_copy = orig_img.copy()
        sketch = Sketcher('input',  [image_copy, mask], lambda: ((255, 255, 255), (255, 255, 255)), opt.thick, opt.painter)

        while True:
            ch = cv2.waitKey()
            if ch == 27:
                print("Quit !")
                break

            elif  ch == ord(" "):
                print("Doing inpainting !")
                # No gradient
                with torch.no_grad():
                    mask_tensor = (ToTensor()(mask))
                    masked_tensor = (img_tensor * (1 - mask_tensor).float()) + mask_tensor
                    first_out, second_out = model(masked_tensor, mask_tensor)
                    comp_tensor = (second_out * mask_tensor + img_tensor * (1 - mask_tensor))
                    
                    pred_np = postProcess(second_out[0])
                    masked_np = postProcess(masked_tensor[0])
                    comp_np = postProcess(comp_tensor[0])

                    cv2.imshow('pred_images', comp_np)
                    print('Inpainting finished!')

            # Reset mask
=======
        
        h, w, c = orig_img.shape
        mask = np.zeros([h, w, 1], np.uint8)
        image_copy = orig_img.copy()
        sketch = Sketcher(
            'input', [image_copy, mask], lambda: ((255, 255, 255), (255, 255, 255)), opt.thick, opt.painter)

        while True:
            ch = cv2.waitKey()
            if ch == ord("q"):
                print("Quit !")
                break

            elif ch == ord(' '):
                print("Inpainting !!!")
                with torch.no_grad():
                    mask_tensor = (ToTensor()(mask)).unsqueeze(0)
                    masked_tensor = (img_tensor * (1 - mask_tensor)) + mask_tensor
                    _, second_out = model(masked_tensor, mask_tensor)
                    complete_img = (second_out * mask_tensor + img_tensor * (1 - mask_tensor))

                    second_np = postProcess(second_out[0])
                    masked_np = postProcess(masked_tensor[0])
                    complete_np = postProcess(complete_img[0])

                    cv2.imshow('Predict_image', complete_img)
                    print('Inpainting finished !!!')

>>>>>>> f37539fad5118af261bc73f00a5adfcf2d343375
            elif ch == ord("r"):
                img_tensor = (ToTensor()(orig_img) * 2.0 - 1.0).unsqueeze(0)
                image_copy[:] = orig_img.copy()
                mask[:] = 0
                sketch.show()
<<<<<<< HEAD
                print("Reset!")

            # Next case
            elif ch == ord("n"):
                print('Next image')
=======
                print("Reset")

            elif ch == ord('n'):
                print("Move to next image")
>>>>>>> f37539fad5118af261bc73f00a5adfcf2d343375
                cv2.destroyAllWindows()
                break

            elif ch == ord("k"):
<<<<<<< HEAD
                print("Keep editing !!")
                img_tensor = comp_tensor
                image_copy[:] = comp_np.copy()
                mask[:] = 0
                sketch.show()
                print("Reset !")
            
=======
                img_tensor = complete_img
                image_copy[:] = complete_np.copy()
                mask[:] = 0
                sketch.show()
                print("Reset")

>>>>>>> f37539fad5118af261bc73f00a5adfcf2d343375
            elif ch == ord("+"):
                sketch.larger_thick()

            elif ch == ord("-"):
<<<<<<< HEAD
                sketch.larger_thick()

            # Save results
            if ch == ord("s"):
                cv2.imwrite(os.path.join(opt.outputs, f'{filename}_masked.png'), masked_np)
                cv2.imwrite(os.path.join(opt.outputs, f'{filename}_pred.png'), pred_np)
                cv2.imwrite(os.path.join(opt.outputs, f'{filename}_comp.png'), comp_np)
                cv2.imwrite(os.path.join(opt.outputs, f'{filename}_mask.png'), mask)

                print('[**] save successfully!')
        cv2.destroyAllWindows()

        if ch == 27:
            break

if __name__ == "__main__":
    demo(opt)
=======
                sketch.small_thick()

            # Save results
            if ch == ord("s"):
                cv2.imwrite(os.path.join(opt.outputs, f"{filename}_masked.png"), masked_np)
                cv2.imwrite(os.path.join(opt.outputs, f'{filename}_pred.png'), second_np)
                cv2.imwrite(os.path.join(opt.outputs, f'{filename}_comp.png'), complete_np)
                cv2.imwrite(os.path.join(opt.outputs, f'{filename}_mask.png'), mask)   

                print("Save image successful")

        cv2.destroyAllWindows()

        if ch == ord("q"):
            break

if __name__=='__main__':
    pass
>>>>>>> f37539fad5118af261bc73f00a5adfcf2d343375

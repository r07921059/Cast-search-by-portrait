import sys,os,dlib,numpy
from skimage import io,transform
import glob
import multiprocessing as mp
import csv
import argparse


select_num = 40

def querySearch(video, query, candi, output_dir, mask_dir):


    fronter_path = "./dlib_model/mmod_human_face_detector.dat"

    predictor_path = "./dlib_model/shape_predictor_68_face_landmarks.dat"

    face_rec_model_path = "./dlib_model/dlib_face_recognition_resnet_model_v1.dat"

    detector = dlib.cnn_face_detection_model_v1(fronter_path)

    sp = dlib.shape_predictor(predictor_path)

    facerec = dlib.face_recognition_model_v1(face_rec_model_path)


    descriptors = []

    candidate = []

    subquery = []

    valids = []

    for f in candi:
        img = io.imread(f)
        h,w,c = img.shape
        dets = detector(img, 1)
        mask_f=os.path.join(mask_dir,video,'candidates',os.path.basename(f).replace('.jpg','_mask.png'))
        mask = io.imread(mask_f) 
        #img[mask==0]=0
        dets = detector(img, 1)
        if len(dets)>1:
            print('masked {}'.format(f))
            img[mask==0]=0
            dets = detector(img, 1)
        max_ = 0
        vec_ = None
        for k, d in enumerate(dets):
            face=d.rect
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            center = (x1+x2)/2.
            #if (x2-x1)*(y2-y1) > max_ and (1./4*w < center< 3./4*w) :
            if (x2-x1)*(y2-y1) > max_:
                shape = sp(img, face)
                face_descriptor = facerec.compute_face_descriptor(img, shape)
                v = numpy.array(face_descriptor)
                vec_ = v
                max_ = (x2-x1)*(y2-y1)


        if max_ > 0:
            candidate.append(f)
            descriptors.append(vec_)
    print('start')
    for f in query:
        img = io.imread(f)
        
        large_pic=False
        h,w,c = img.shape
        print('f:{}'.format(img.shape))
        
        img =255* transform.resize(img,(int(2/3.*h),int(2/3.*w)))
        img = img.astype(numpy.uint8)
        while(img.shape[0]>=3000 or img.shape[1]>=3000):
            h,w,c = img.shape
            img =255* transform.resize(img,(int(2/3.*h),int(2/3.*w)))
            img = img.astype(numpy.uint8)
        dets = detector(img, 1)

        dist = []
        
        if len(dets) == 0:
    #    dets.append(dlib.mmod_rectangles(0,0,img.shape[0]-1,img.shape[1]-1))
        
            print(img.shape)
            print(f)
            shape = sp(img, dlib.rectangle(0,0,img.shape[0]-1,img.shape[1]-1))
        
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            d_test = numpy.array(face_descriptor)
            for i in descriptors:
                dist_ = numpy.linalg.norm(i -d_test)
                dist.append(dist_)
        
        else:
            max_=0
            vec_ = None
            for k, d in enumerate(dets):
                face = d.rect
                print(face)
                if large_pic:
                    print(face)
                    x1 = int(face.left()*3/2)
                    y1 = int(face.top()*3/2)
                    x2 = int(face.right()*3/2)
                    y2 = int(face.bottom()*3/2)
                    face=dlib.rectangle(x1,y1,x2,y2)
                    print(face)
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                if (x2-x1)*(y2-y1) > max_:
                    shape = sp(img, face)
                    face_descriptor = facerec.compute_face_descriptor(img, shape)
                    v = numpy.array(face_descriptor)
                    vec_ = v
                    max_ = (x2-x1)*(y2-y1)


            if max_ > 0:
                d_test = vec_#numpy.array(face_descriptor)
                for i in descriptors:
                    dist_ = numpy.linalg.norm(i -d_test)
                    dist.append(dist_)

        c_d = dict(zip(candidate,dist))

        cd_sorted = sorted(c_d.items(), key = lambda d:d[1])
        
        tmp=sorted(dist)
        # print(tmp)
        valid = sum(float(k)<0.55  for k in tmp)
        if valid>select_num:
            valid = select_num
        elif valid==0:
            valid = 5
        
        print(valid)
        
        for i in range(select_num):
            valids.append(valid)
            subquery.append(cd_sorted[i][0])
    print("finish %s" % video)
    with open(os.path.join(output_dir,'{}.txt'.format(video)), 'w') as f:
        f.write('\n'.join(subquery))
    print(valids)
    with open(os.path.join(output_dir,'{}_valid.txt'.format(video)), 'w') as f:
        for item in valids:
            f.write("{}\n".format(item))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default='./reproduce_tcnn_mask_subquery', type=str, help="destination of output log file")
    parser.add_argument("--video_dir", type=str, default='./final_data/val', help="where video lies e.g. ./final_data/val")
    parser.add_argument("--mask_dir", type=str, default='./final_data_mask/val', help="where mask lies e.g. ./final_data_mask/val")
    opt = parser.parse_args()
    videos = sorted(os.listdir(opt.video_dir))
    #pool = mp.Pool(processes = mp.cpu_count()-4)
    true_output_dir = os.path.join(opt.output_dir,os.path.basename(opt.video_dir.strip('/')))
    os.makedirs(true_output_dir, exist_ok=True)


    for i,v in enumerate(videos):
        print(v)
        query = os.path.join(opt.video_dir,v,'cast')
        candi = os.path.join(opt.video_dir,v,'candidates')
        queryImg = sorted(glob.glob(os.path.join(query, '*.jpg')))
        candImg = sorted(glob.glob(os.path.join(candi, '*.jpg')))
        subquery = querySearch(v,queryImg, candImg,true_output_dir, opt.mask_dir)

        #pool.apply_async(querySearch, args=(v, queryImg, candImg,))
    #pool.close()
    #pool.join()

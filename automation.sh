# unet-static.json
# python src/python/train.py --config configs/unet-static.json
# python src/python/evaluate.py --config configs/unet-static.json

# # unet.json
# python src/python/train.py --config configs/unet.json
# python src/python/evaluate.py --config configs/unet.json
# pmnetv3.json
python src/python/train.py --config configs/pmnetv3.json
python src/python/evaluate.py --config configs/pmnetv3.json
# transunet.json
python src/python/train.py --config configs/transunet.json
python src/python/evaluate.py --config configs/transunet.json

#from mydataset import SquadDataset

#dataset = SquadDataset()
#dataloader = DataLoader(dataset)


from datasets import load_dataset
datasets = load_dataset('squad')
print(datasets)


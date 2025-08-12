
files = ["LipCoordFormer_3DConv_Coord_BiGRU_training_log_20250512114337.txt"]
file_lens = [410]

total_val_losses = []
total_cers = []
total_wers = []

for idx in range(0, len(files)):
    val_losses = []
    cers = []
    wers = []
    epochs = 0
    
    file = open(files[idx], "r")
    lines = file.readlines()
    print(len(lines))
    for i in range(0, len(lines)):
        line = lines[i]
        if "] Validation - " in line:
            # next_line = lines[i+1]
            # print(next_line)
            # values = next_line.split(" - ")[1]
            values = line.split("Validation - ")[1]
            print(values)
            val_loss, cer, wer = values.split(",")
            val_loss = val_loss.split(": ")[1].strip()
            cer = cer.split(": ")[1].strip()
            wer = wer.split(": ")[1].strip()
            val_losses.append(val_loss + ",")
            cers.append(cer + ",")
            wers.append(wer + ",")
            epochs += 1

            if epochs == file_lens[idx]:
                print("epochs: ", epochs)
                break

    # print(len(val_losses))
    # print(len(cers))
    # print(len(wers))

    total_val_losses.extend(val_losses)
    total_cers.extend(cers)
    total_wers.extend(wers)

print(len(total_val_losses))
print(len(total_cers))
print(len(total_wers))  

write_file = "./loss-curves/table3-3DConv+Coord+BiGRU+NoTransfomer.txt"

with open(write_file, "a") as file:
    file.write("val_losses=[")
    file.writelines(total_val_losses)
    file.write("] \n\ncers=[")
    file.writelines(total_cers)
    file.write("] \n\nwers=[")
    file.writelines(total_wers)
    file.write("]")

    
    
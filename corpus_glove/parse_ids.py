input_file1 = './data/friends.txt.ids20000'
input_file2 = './data/opensubtitles.txt.ids20000'
output_file = './data/op+fri.txt.ids20000'

limit = 28

with open(input_file1, 'r') as f1:
    with open(input_file2, 'r') as f2:
        with open(output_file, 'w') as f3:
            for line in f1.readlines()+f2.readlines():
                processed_line = ''
                for char in line:
                    processed_line += char
                linelist = processed_line.split()
                if len(linelist) < limit:
                    linelist.insert(0, '1')
                    linelist.append('2')
                    for i in range(30 - len(linelist)):
                        linelist.append('0')
                    output_line = ' '.join(linelist)
                    f3.write(output_line)
                    f3.write('\n')

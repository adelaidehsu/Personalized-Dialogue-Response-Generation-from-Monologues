input_file1 = './data/friends.txt'
input_file2 = './data/opensubtitles.txt'
output_file = './data/op+fri.txt'

limit = 28
mark = [',', '.', '!', '?', ':', ';']

with open(input_file1, 'r') as f1:
    with open(input_file2, 'r') as f2:
        with open(output_file, 'w') as f3:
            for line in f1.readlines()+f2.readlines():
                processed_line = ''
                for char in line:
                    if char in mark:
                        processed_line += ' '
                    processed_line += char
                linelist = processed_line.split()
                if len(linelist) < limit:
                    linelist.insert(0, '<GO>')
                    linelist.append('<EOS>')
                    for i in range(30 - len(linelist)):
                        linelist.append('<PAD>')
                    output_line = ' '.join(linelist)
                    f3.write(output_line)
                    f3.write('\n')

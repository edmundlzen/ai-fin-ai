import camelot

test_file = "pb_phs/phs_PB ASIA EQUITY FUND.pdf"

tables = camelot.read_pdf(test_file)

print(tables)

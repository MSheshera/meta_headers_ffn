"""
Creates the simstring databases used for the approximate string matching.
"""
import os, sys
import simstring

dict_dir = '/home/msheshera/MSS/Code/Projects/meta_headers/feature_dicts/dicts'
out_dir = './simstringdb'
people = ['people_frequent_last_names.txt', 'people_shared.txt',
		'people_english_only.txt', 'people_chinese_only.txt']
places = ['city_full.txt', 'country_full.txt', 'region_full.txt']
departments = ['department_full.txt', 'department_keywords.txt', 
'faculty_full.txt', 'faculty_keywords.txt']
universities = ['institution_full.txt', 'institution_keywords.txt', 'university_full.txt', 'university_keywords.txt']            

# Create databases.
def create_dbs():
	"""
	Reads in the files specified in the global lists and creates simstring
	databases.
	"""
	for name, fnames in [('people', people), ('places', places), 
		('departments', departments), ('universities', universities)]:
		out_dbname = os.path.join(out_dir, name+'.db')
		# Enable creating the database in unicode mode.
		group_db = simstring.writer(out_dbname,  3, False, True)
		for fname in fnames:
			fname = os.path.join(dict_dir, fname)
			with open(fname, 'r') as file:
				for line in file:
					group_db.insert(line.strip())
		group_db.close()
		print 'Wrote: ', out_dbname

# Test out reading from the database
def test_matches():
	dbs = ['people.db', 'places.db', 'departments.db', 'universities.db']
	for dbname in dbs:
		db = simstring.reader(os.path.join(out_dir, dbname))
		db.measure = simstring.cosine
		db.threshold = 0.6
		print(db.retrieve(u'london'.encode('utf-8')))

if __name__ == '__main__':
	create_dbs()
	test_matches()
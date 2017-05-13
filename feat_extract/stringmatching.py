"""
This is just come code to create the simstring databases when pointed to the 
text lexicons. 
"""
import os, sys, errno
import simstring
import fe_settings

def create_dbs():
    """
    Reads in the files specified in the lists specified in fe_settings and 
    creates simstring databases.
    """
    for name, fnames in [('people', fe_settings.people), 
        ('places', fe_settings.places), 
        ('departments', fe_settings.departments), 
        ('universities', fe_settings.universities)]:
        out_dbname = os.path.join(fe_settings.simstringdb_dir, name+'.db')
        # Enable creating the database in unicode mode.
        group_db = simstring.writer(out_dbname,  3, False, True)
        for fname in fnames:
            fname = os.path.join(fe_settings.lexicon_dir, fname)
            with open(fname, 'r') as file:
                for line in file:
                    group_db.insert(line.strip())
        group_db.close()
        print 'Wrote: ', out_dbname

def test_matches():
    """
    Just tests reading from the databases.
    """
    dbs = ['people.db', 'places.db', 'departments.db', 'universities.db']
    for dbname in dbs:
        db = simstring.reader(os.path.join(fe_settings.simstringdb_dir, dbname))
        db.measure = simstring.cosine
        db.threshold = 0.6
        print(db.retrieve(u'london'.encode('utf-8')))

if __name__ == '__main__':
    # Create output direcory if it doesnt exist.
    try:
        os.makedirs(fe_settings.simstringdb_dir)
        print('Created {} for simstring databases'.format(
            fe_settings.simstringdb_dir))
        sys.stdout.flush()
    except OSError as ose:
        # For the case of *file* by name of simstringdb_dir existing
        if (not os.path.isdir(fe_settings.simstringdb_dir)) and (ose.errno == errno.EEXIST):
            sys.stderr.write('IO ERROR: Could not create output directory\n')
            sys.exit(1)
        # If its something else you don't know; report it and exit.
        if ose.errno != errno.EEXIST:
            sys.stderr.write('OS ERROR: {:d}: {:s}: {:s}\n'.format(ose.
                errno, ose.strerror, fe_settings.simstringdb_dir))
            sys.stdout.flush()
            sys.exit(1)
    create_dbs()
    test_matches()
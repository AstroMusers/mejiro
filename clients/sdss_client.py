def query_builder(fields, table, conditions):

    if conditions is not None:
        return 'SELECT ' + ', '.join(fields) + \
            ' FROM ' + table + \
            ' WHERE ' + ' AND '.join(conditions)
    else:
        return 'SELECT ' + ', '.join(fields) + \
            ' FROM ' + table

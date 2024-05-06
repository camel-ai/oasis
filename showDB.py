# File: showDB.py
from prettytable import PrettyTable
from sqlalchemy import create_engine, MetaData, select, inspect

class TwitterDB:
    def __init__(self, db_uri='sqlite:///twitter_simulation.db'):
        self.engine = create_engine(db_uri, echo=True)
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)

    def print_db_tables_info(self):
        """Prints details of all tables in the database."""
        for table_name in self.metadata.tables:
            print(f"Table: {table_name}")
            table = PrettyTable()
            # Extended columns list to include 'Default', 'Indexes', 'Foreign Keys', and 'Unique'
            table.field_names = ["Column Name", "Type", "Nullable", "Primary Key", "Default", "Indexes", "Foreign Keys", "Unique"]
            columns = self.metadata.tables[table_name].columns
            for column in columns:
                # Gather index information
                indexes = [index.name for index in column.table.indexes if column in index.columns]
                # Gather foreign key information
                fkeys = [str(fk.column) for fk in column.foreign_keys]
                # Check for unique constraints
                unique = any(column.name in [c.name for c in constraint.columns] and constraint.__visit_name__ == 'unique_constraint' for constraint in column.table.constraints)
                # Add row to table with all these details
                table.add_row([
                    column.name, str(column.type), column.nullable, column.primary_key,
                    str(column.default), ', '.join(indexes), ', '.join(fkeys), unique
                ])
            print(table)
            print("\n")  # Add a space between tables for readability

    def print_table_data(self, table_name):
        """Prints all data from a specified table."""
        table = self.metadata.tables.get(table_name)
        if table is not None:
            query = select(table.c)
            with self.engine.connect() as connection:
                result = connection.execute(query)
                data_table = PrettyTable()
                data_table.field_names = [column.name for column in table.columns]
                for row in result:
                    data_table.add_row(row)
                print(data_table)
        else:
            print(f"No table found with the name {table_name}")

    def print_all_tables_data(self):
        """Prints data for all tables in the database."""
        for table_name in self.metadata.tables:
            print(f"Data for table: {table_name}")
            self.print_table_data(table_name)
            print("\n")  # Add a space between tables for readability

# Example usage
db = TwitterDB()
db.print_db_tables_info()
db.print_all_tables_data()  # Print data from all tables

class Employee:
    raise_amount = 1.04
    no_of_emp = 0

    def __init__(self, first, second, pay):
        self.first_name = first
        self.second_name = second
        self.pay = pay
        self.full_name = '{}:{}'.format(first, second)

        Employee.no_of_emp += 1

    def name_pay(self):
        return '{}/{}'.format(self.full_name, self.pay)

    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amount)

    @classmethod
    def from_string(cls, str):
        first, second, pay = str.split('-')
        return cls(first, second, pay)



emp_1 = Employee('Eimon', 'Eimran', 50000)
emp_2 = Employee('Test_1', 'Test_2', 10000)
emp_str = 'AB-DC-1000'

emp_3 = Employee.from_string(emp_str)

print(emp_3.name_pay())

function tests = test_example
%TEST_EXAMPLE Example test suite for MATLAB unit testing
%
%   TESTS = TEST_EXAMPLE() creates a test suite using functiontests framework.
%
%   This is a simple example test suite demonstrating basic unit testing
%   functionality in MATLAB.
%
%   Example:
%       tests = test_example;
%       run(tests);
%
%   See also: functiontests, localfunctions

arguments
end

tests = functiontests(localfunctions);
end

function test_truth(testCase)
%TEST_TRUTH Test basic arithmetic truth
%
%   TEST_TRUTH(TESTCASE) verifies that basic arithmetic operations work correctly.
%
%   Input Arguments:
%   ----------------
%   TESTCASE - Test case object from functiontests framework

arguments
    testCase (1,1) matlab.unittest.TestCase
end

verifyEqual(testCase, 1+1, 2);
end
